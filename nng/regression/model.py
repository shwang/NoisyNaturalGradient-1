from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import typing
from typing import Iterable, List, Optional, Tuple, Sequence

import tensorflow as tf
from tensorflow.contrib.framework import with_shape

from nng.core.base_model import BaseModel
from nng.misc.registry import get_model
from nng.regression.controller.bayesian_learning import BayesianLearning
from nng.regression.misc.layers import *
from nng.regression.controller.sample import NormalOutSample
from nng.regression.network.ffn import *

if typing.TYPE_CHECKING:
    from irdplus.bnn.problem.problem import Problem


class Model(BaseModel):
    def __init__(self, config, input_dim: Iterable[int], n_data: int, *,
            n_particles_ph: Optional[tf.Tensor] = None,
            inputs_ph: Optional[tf.Tensor] = None,
            problem: "Optional[Problem]" = None):
        """ Initialize a class Model.
        :param config: Configuration Bundle.
        :param input_dim: int
        :param n_data: int
        """
        super().__init__(config)
        # Set the approximation type specifically.
        if config.optimizer == "ekfac":
            self.layer_type = "emvg"
        elif config.optimizer == "kfac":
            print("[!] Optimizer: KFAC")
            self.layer_type = "mvg"
        else:
            print("[!] Optimizer: {}".format(config.optimizer))
            self.layer_type = None
        self.input_dim = input_dim  # type: List[int]
        self.n_data = n_data  # type: int
        self.problem = problem

        # Initialize attributes.
        self.n_particles = n_particles_ph or tf.placeholder(tf.int32)  # type: tf.Tensor
        inputs_shape = [self.n_particles] + list(self.input_dim)
        self.inputs = with_shape(tf.convert_to_tensor(inputs_shape),
                inputs_ph or tf.placeholder(tf.float32,
                        shape=[None] + list(self.input_dim)))  # type: tf.Tensor

        # self.is_training = None  ##
        self.targets = tf.placeholder(tf.float32, [None])  # type: tf.Tensor
        self.alpha = tf.placeholder(tf.float32, shape=[], name='alpha')  # type: tf.Tensor
        self.beta = tf.placeholder(tf.float32, shape=[], name='beta')  # type: tf.Tensor
        self.omega = tf.placeholder(tf.float32, shape=[], name='omega')  # type: tf.Tensor

        if not self.problem:
            self.stub = "regression"
        else:
            self.stub = self.problem.stupid_stub

        # Build the model.
        self._build_model()
        self.init_saver()

    def _build_model(self):
        net = get_model(self.config.model_name)

        layers, init_ops, num_hidden = net(self.layer_type,
                                           int(self.inputs.shape[-1]),
                                           self.n_data,
                                           self.config.kl,
                                           self.config.eta,
                                           self.alpha,
                                           self.beta,
                                           self.config.damping,
                                           self.omega)

        outsample_cls = NormalOutSample if self.stub == "regression" else None
        if self.layer_type == "emvg":
            self.learn = BayesianLearning(
                    layer_sizes=[self.inputs.shape[-1], num_hidden, 1],
                    layer_types=[EMVGLayer] * 2,
                    layer_params=[{}] * 2,
                    out_params={},
                    activation_fn=tf.nn.relu,
                    outsample_cls=outsample_cls,
                    x=self.inputs,
                    y=self.targets,
                    n_particles=self.n_particles,
                    std_y_train=self.config.std_train,
                    stub=self.stub)
        elif self.layer_type == "mvg":
            self.learn = BayesianLearning(
                    layer_sizes=[self.inputs.shape[-1], 50, 1],
                    layer_types=[MVGLayer] * 2,
                    layer_params=[{}] * 2,
                    out_params={},
                    activation_fn=tf.nn.relu,
                    outsample_cls=outsample_cls,
                    x=self.inputs,
                    y=self.targets,
                    n_particles=self.n_particles,
                    std_y_train=self.config.std_train,
                    stub=self.stub)
        else:
            raise ValueError(self.layer_type)

        self._log_py_xw = self._build_log_py_xw()
        self.kl = self.learn.build_kl()
        self.loss_prec = self._build_loss_prec()
        self.lower_bound = self._build_lower_bound()

        self.mean_log_py_xw = tf.reduce_mean(self._log_py_xw)
        self.rmse = self.learn.rmse
        self.ll = self.learn.log_likelihood
        self.h_pred = self.learn.h_pred

        optimizer = tf.train.AdamOptimizer(self.config.learning_rate)

        self.init_ops = tf.group(init_ops) if init_ops != [] else None
        weight_update_op, self.basis_update_op, self.scale_update_op = \
                self._build_layer_update_ops(layers)
        prec_op = self._build_prec_op()
        self.train_op = tf.group([weight_update_op, prec_op], name="train_op")

    def get_train_output(self) -> tf.Tensor:
        """
        Returns the output that should be used for calculating training loss.
        """
        if self.stub == "regression":
            return self.learn._y
        elif self.stub == "ird":
            return self.h_pred
        else:
            raise ValueError(self.stub)

    def get_test_output(self) -> tf.Tensor:
        """
        Returns the output that should be used for testing.
        """
        if self.stub == "regression":
            return self.h_pred
        elif self.stub == "ird":
            return self.h_pred
        else:
            raise ValueError(self.stub)

    def init_saver(self):
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)

    def _build_log_py_xw(self) -> tf.Tensor:
        if self.stub == "regression":
            return self.learn.log_py_xw
        elif self.stub == "ird":
            output = self.get_train_output()
            return self.problem.build_data_loss(output,
                    l1_loss=0.0, l2_loss=0.0)
        else:
            raise ValueError(self.problem.stupid_stub)

    def _build_loss_prec(self) -> tf.Tensor:
        if self.stub != "regression":
            return tf.constant(0.0, name="loss_prec")

        # log_alpha, log_beta = tf.trainable_variables()[-2:]  # Ewww
        outsample = self.learn._net._outsample
        alpha_ = outsample.q_alpha
        beta_ = outsample.q_beta
        # alpha_ = tf.exp(log_alpha)
        # beta_ = tf.exp(log_beta)
        y_obs = tf.tile(tf.expand_dims(self.targets, 0), [self.n_particles, 1])
        h_pred = tf.squeeze(self.learn.h_pred, 2)
        loss_prec = 0.5 * (tf.stop_gradient(tf.reduce_mean((y_obs - h_pred) ** 2)) *
                           alpha_ / beta_ - (tf.digamma(alpha_) - tf.log(beta_ + 1e-10)))
        return loss_prec

    def _build_lower_bound(self) -> tf.Tensor:
        lower_bound = tf.reduce_mean(
                self._log_py_xw - \
                self.config.kl * self.kl / self.n_data)
        lower_bound = lower_bound - tf.reduce_mean(self.loss_prec)

        return lower_bound

    def _build_prec_op(self) -> tf.Operation:
        """Build an operation used to update outsampling precision."""
        outsample = self.learn._net._outsample
        optimizer = tf.train.AdamOptimizer(self.config.learning_rate)
        prec_vars = [outsample.prec_logalpha, outsample.prec_logbeta]
        prec_grads = optimizer.compute_gradients(-self.lower_bound, prec_vars)
        prec_op = optimizer.apply_gradients(prec_grads)
        return prec_op

    def _build_layer_update_ops(self, layers: Sequence) -> Tuple[
            tf.Operation, tf.Operation, tf.Operation]:
        """
        Builds the following layer Operations:
        `weight_update_op, scale_update_op, basis_update_op`.
        """
        qws = [self.learn.qws['w' + str(i)].tensor
                for i in range(len(self.learn.qws))]
        w_grads = tf.gradients(tf.reduce_sum(
            tf.reduce_mean(self._log_py_xw, 1), 0), qws)

        activations = [get_collection("a0"), get_collection("a1")]
        activations = [tf.concat(
            [activation,
             tf.ones(tf.concat([tf.shape(activation)[:-1], [1]], axis=0))], axis=-1) for activation in activations]

        s = [get_collection("s0"), get_collection("s1")]
        if self.config.true_fisher:
            # Take gradient with y=target and the rest of the model sampled.
            sampled_log_prob = self.learn.sampled_log_prob
            s_grads = tf.gradients(tf.reduce_sum(sampled_log_prob), s)
        else:
            # Take gradient with y sampled along with the model.
            s_grads = tf.gradients(tf.reduce_sum(self._log_py_xw), s)
        # XXX: I believe that for IRD, not true_fisher is meaningless.
        # Because there is no target, we can only sample "y".

        weight_updates = []
        scale_updates = []
        basis_updates = []
        for l, w, w_grad, a, s_grad in zip(layers, qws, w_grads, activations, s_grads):
            # Adds the regular KFAC update.
            weight_updates.extend(l.update(w, w_grad, a, s_grad))
            if self.layer_type == "emvg":
                scale_updates.extend(l.update_scale(w, w_grad, a, s_grad))
                basis_updates.extend(l.update_basis(w, w_grad, a, s_grad))

        return (tf.group(*weight_updates, name="weight_updates"),
                tf.group(*scale_updates, name="scale_updates"),
                tf.group(*basis_updates, name="basis_updates"))
