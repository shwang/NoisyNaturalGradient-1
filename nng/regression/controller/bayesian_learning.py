from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from typing import Optional, Type

import numpy as np
import tensorflow as tf
import zhusuan as zs

from nng.regression.controller.sample import NormalOutSample
from nng.regression.misc.layers import FeedForward
from nng.regression.misc.eval_utils import rmse, log_likelihood
from nng.regression.misc.collections import add_to_collection


class BayesianNetwork(object):
    """ BayesianNetwork is a class with flexible priors and variational posteriors.
    """
    def __init__(self, layer_sizes, layer_types, layer_params, out_params,
            activation_fn,
            outsample_cls: "Optional[Type[NormalOutSample]]" = None):
        """ Initialize BayesianNetwork.
        :param layer_sizes: [int]
        :param layer_types: [Layer]
        :param layer_params: [dict]
        :param out_params: [dict]
        :param activation_fn: activation function
        :param outsample_cls: Type[OutSample]
        """
        super(BayesianNetwork, self).__init__()
        self._layer_sizes = layer_sizes
        self._layer_types = layer_types
        self._layer_params = layer_params
        self._out_params = out_params
        self._activation_fn = activation_fn
        self.layers = []
        self.stochastic_names = []
        self._kl_weights = []
        if not ((len(layer_sizes) == len(layer_types) + 1) and (len(layer_types) == len(layer_params))):
            raise ValueError('length of layer_type, layer_params, layer_sizes must be compatible')
        for i, (n_in, n_out, layer_type, params) in enumerate(zip(layer_sizes[:-1],
                                                              layer_sizes[1:],
                                                              layer_types,
                                                              layer_params)):
            self.layers.append(layer_type(n_in, n_out, 'w'+str(i), params))
            self.stochastic_names.append('w'+str(i))
        self._kl_weights.extend(self.layers)

        if outsample_cls is None:
            self._outsample = None
        else:
            self._outsample = outsample_cls(self.out_params)
            self.stochastic_names += self._outsample.stochastic_names
            self._kl_weights.append(self._outsample)

        self.first_build = True

    def build_kl(self):
        kl = 0.
        for w in self._kl_weights:
            kl = kl + w.kl_exact
        return kl

    @property
    def layer_sizes(self):
        return self._layer_sizes

    @property
    def layer_types(self):
        return self.layer_types

    @property
    def layer_params(self):
        return self._layer_params

    @property
    def out_params(self):
        return self._out_params

    @property
    def activation_fn(self):
        return self._activation_fn

    @property
    def input_size(self):
        return self.layer_sizes[0]

    @property
    def output_size(self):
        return self.layer_sizes[-1]

    @property
    def outsample(self):
        return self._outsample

    @property
    def n_weights(self):
        n = 0
        for n_in, n_out in zip(self.layer_sizes[:-1], self.layer_sizes[1:]):
            n = n + (n_in+1) * n_out
        return n

    def _forward(self, inputs, n_particles):
        """ Forward the inputs through the network, sampling the weights
            from the prior distribution, not the variational distribution.
        :param inputs: tensor of shape [batch_size, n_x] (n_x = self.layer_sizes[0])
        :param n_particles: tensor. Number of samples.
        :return: tensor of shape [n_particles, batch_size]
        """
        h = tf.tile(tf.expand_dims(inputs, 0), [n_particles, 1, 1])
        for i, l in enumerate(self.layers[:-1]):
            if self.first_build:
                add_to_collection('a'+str(i), h)
            h = l.forward(h)
            if self.first_build:
                add_to_collection('s'+str(i), h)
            h = self.activation_fn(h)
        l = self.layers[-1]
        if self.first_build:
            add_to_collection('a'+str(i+1), h)
        h = l.forward(h)
        if self.first_build:
            add_to_collection('s'+str(i+1), h)
        return h

    def predict(self, inputs, n_particles):
        """ Forward the inputs through the network and get the outputs. (Weights
            are sampled from the prior distribution, not the variational
            distribution.
        :param inputs: tensor of shape [batch_size, n_x] (n_x = self.layer_sizes[0])
        :param n_particles: tensor. Number of samples.
        :return output: tensor of shape [n_particles, batch_size]. A Gaussian
          sample with mean equal to the neural network.
        :return h: tensor of shape [n_particles, batch_size, 1]. Final hidden layer.
        """
        h = self._forward(inputs, n_particles)
        if self._outsample:
            output = self._outsample.forward(h)
        else:
            output = h
        if self.first_build:
            self.first_build = False
        return output, h


class BayesianLearning(object):
    """ A class to learn BNN.
    """
    def __init__(self, layer_sizes, layer_types, layer_params, out_params, activation_fn, outsample,
                 x, y, n_particles, **kwargs):
        """ Initialize class BayesianLearning.
        """

        self._net = BayesianNetwork(layer_sizes, layer_types, layer_params, out_params,
                                    activation_fn, outsample)
        self._n_particles = n_particles
        self._x = x
        self._y = y

        # Holds all the string keys of variational weights. Seems to be
        # the same as `list(self._qws)`, with reordering.
        self._q_names = []
        # Holds all the parameters of the variational distribution.
        self._qws = {}  # Dict[String -> StochasticTensor].
        with zs.BayesianNet() as variational:
            for l in self._net.layers:
                if isinstance(l, FeedForward):
                    self._q_names.append(l.w_name)
                    # Calling .qws() builds and returns a StochasticTensor
                    # in this BayesNet context corresponding to somehting.
                    self._qws.update({l.w_name: l.qws(self.n_particles)})

            if self._net._outsample and hasattr(self._net._outsample, 'qs'):
                self._q_out = self._net._outsample.qs(self.n_particles)
                self._q_names = self._q_names + self._net._outsample.stochastic_names[:-1]

                if not isinstance(self._q_out, list):
                    self._q_out = [self._q_out]
        self._variational = variational

        # observed dict
        if hasattr(self, '_q_out'):
            self._qs = zs.merge_dicts(
                self._qws,
                dict(zip(self._net.outsample.stochastic_names[:-1], self._q_out)))
        else:
            self._qs = self._qws

        # `self._qs` is now a Dictionary that maps each StochasticTensor name in
        # `variational` to its StochasticTensor. Perfect for building an
        # observation dictionary where everything variational parameter is
        # observed.

        """
        In the case of Regression problem,
           * y_pred != h_pred. (Since we outsample y_pred from a Normal
                                      distribution).
           * Also, by default we should observe {'y': tiled(target)}.
           * log_py_xw and sampled_log_prob come from evaluating zs log prob
                    of model.get("y"), (either observed y or sampled y).
        In the case of IRD problem,
           * y_pred == h_pred? (Since there is no outsample before loss)
           * log_py_xw and sampled_log_prob come from evaluating a Tensor
                   representing IRD loss. We should still observe model
                   parameters as variational samples.
        """

        @zs.reuse('buildnet')
        def buildnet(observed):
            """ Get the BayesianNet instance and output of the bayesian neural network with some
            nodes observed.
            :param observed: dict of (str, tensor). Representing the mapping from node name to value.
            :return: BayesianNet instance.
            """
            with zs.BayesianNet(observed=observed) as model:
                y_pred, h_pred = self._net.predict(self.x, self.n_particles)
            return model, y_pred, h_pred
        self.buildnet = buildnet

        y_obs = tf.tile(tf.expand_dims(self.y, 0), [self.n_particles, 1])
        # BayesianNet instance with every stochastic node observed.
        model, dist, _ = self.buildnet(zs.merge_dicts(self._qs, {'y': y_obs}))
        self._model = model
        self._dist = dist  # = model.outputs("y")
        self._kwargs = kwargs

    @property
    def variational(self):
        return self._variational

    @property
    def model(self):
        return self._model

    @property
    def dist(self):
        return self._dist

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    @property
    def n_particles(self):
        """number of samples"""
        return self._n_particles

    @property
    def q_names(self):
        """node names of weights"""
        return self._q_names

    @property
    def log_py_xw(self):
        """ Log likelihood of output.
        :return: tensor of shape [n_particles, batch_size]
        """
        return self.model.local_log_prob('y')

    @property
    def sampled_log_prob(self):
        targets = self.dist.sample(1)
        log_prob = tf.reduce_mean(self.dist.log_prob(tf.stop_gradient(targets)), 0)
        return log_prob

    def build_kl(self) -> tf.Tensor:
        """ KL divergence of the variational posterior over the layer weights
        from the weights prior.
        :return: tensor of shape [n_particles]
        """
        return self._net.build_kl()

    @property
    def h_pred(self):
        """ Final hidden layer of the network.
        :return: tensor of shape [n_particles, batch_size, n]
        """
        _, _, h_pred = self.buildnet(self._qs)
        return h_pred

    @property
    def y_pred(self):
        """ Return Output.
        :return: tensor of shape [n_particles, batch_size]
        """
        _, y_pred, _ = self.buildnet(self._qs)
        return y_pred

    @property
    def kwargs(self):
        """ Useful info for the class.
        :return: dict
        """
        return self._kwargs

    @property
    def qs(self):
        """ The observed dict with all nodes except output.
        For example, including 'w':qw, 'y_prec':q_prec
        :return: dict of (str, tensor).
        """
        return self._qs

    @property
    def qws(self):
        return self._qws

    @property
    def rmse(self):
        if not self._net._outsample or not self._net._outsample.task == 'regression':
            return np.nan
        y_pred = tf.reduce_mean(self.h_pred, [0, 2])
        return rmse(y_pred, self.y, self.kwargs['std_y_train'])

    @property
    def log_likelihood(self):
        if not self._net._outsample or not self._net._outsample.task == 'regression':
            return np.nan
        return log_likelihood(self.log_py_xw, self.kwargs['std_y_train'])
