from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import typing
from typing import Iterable, List, Optional, Tuple, Sequence

import tensorflow as tf

from nng.core.base_model import BaseModel
from nng.misc.registry import get_model
from nng.regression.controller.bayesian_learning import BayesianLearning
from nng.regression.misc.layers import *
from nng.regression.controller.sample import NormalOutSample
from nng.regression.network.ffn import *

if typing.TYPE_CHECKING:
    from irdplus.bnn.problem.problem import Problem


class Model(BaseModel):

    n_particles = ...  # type: tf.Tensor
    inputs_shape = ... # type: tf.Tensor
    inputs = ... # type: tf.Tensor
    targets = ... # type: tf.Tensor
    alpha = ... # type: tf.Tensor
    beta = ... # type: tf.Tensor
    omega = ... # type: tf.Tensor

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
        if n_particles_ph is None:
            self.n_particles = tf.placeholder(tf.int32)
        else:
            self.n_particles = n_particles_ph

        if inputs_ph is None:
            inputs_ph = tf.placeholder(tf.float32,
                        shape=[None] + list(self.input_dim))
        self.inputs = inputs_ph

        self.targets = tf.placeholder(tf.float32, [None])
        self.alpha = tf.placeholder(tf.float32, shape=[], name='alpha')
        self.beta = tf.placeholder(tf.float32, shape=[], name='beta')
        self.omega = tf.placeholder(tf.float32, shape=[], name='omega')

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

        self.h_pred = self.learn.h_pred
        if self.stub == "ird":
            self._main, self._aux, self._bnn = \
                    self.problem.gather_standard_rewards(self.h_pred)
        else:
            self._main = self._aux = self._bnn = None

        self._log_py_xw = self._build_log_py_xw()
        self.kl = tf.check_numerics(self.learn.build_kl(), "kl")
        self.loss_prec = self._build_loss_prec()
        self.lower_bound = self._build_lower_bound()

        self.mean_log_py_xw = self._log_py_xw
        self.rmse = self.learn.rmse
        self.ll = self.learn.log_likelihood

        optimizer = tf.train.AdamOptimizer(self.config.learning_rate)

        self.init_ops = tf.group(init_ops) if init_ops != [] else None
        weight_update_op, self.basis_update_op, self.scale_update_op = \
                self._build_layer_update_ops(layers)
        prec_op = self._build_prec_op()
        self.train_op = tf.group([weight_update_op, prec_op], name="train_op")

    def sample_outputs(self, feat, n_samples):
        fd = { self.model.inputs: feat,
               self.model.n_particles: n_samples,}
        return self.sess.run(self.model.h_pred, feed_dict=fd)

    def sample_test_rewards(self, feat, n_samples):
        assert self.stub == "ird"
        fd = self.model.problem.test_fd()
        fd[self.model.n_particles] = n_samples
        fetches = [self.model._bnn, self.model._main]
        return self.sess.run(fetches, feed_dict=fd)

    def get_train_output(self) -> tf.Tensor:
        """
        Returns the output that should be used for calculating training loss.
        """
        if self.stub == "regression":
            return self.learn.y_pred
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
            # return self.learn.log_py_xw
            return tf.reduce_sum(self.learn.log_py_xw)
        elif self.stub == "ird":
            output = self.get_train_output()
            loss = self.problem.build_data_loss(output,
                    l1_loss=0.0, l2_loss=0.0)
            return -loss
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

        return tf.check_numerics(lower_bound, "lb")

    def _build_prec_op(self) -> tf.Operation:
        """Build an operation used to update outsampling precision."""
        if self.stub == "regression":
            outsample = self.learn._net._outsample
            optimizer = tf.train.AdamOptimizer(self.config.learning_rate)
            prec_vars = [outsample.prec_logalpha, outsample.prec_logbeta]
            prec_grads = optimizer.compute_gradients(-self.lower_bound,
                    prec_vars)
            prec_op = optimizer.apply_gradients(prec_grads, name="prec_op")
        else:
            prec_op = tf.no_op(name="prec_op")
        return prec_op

    def _build_layer_update_ops(self, layers: Sequence) -> Tuple[
            tf.Operation, tf.Operation, tf.Operation]:
        """
        Builds the following layer Operations:
        `weight_update_op, scale_update_op, basis_update_op`.
        """
        qws = [self.learn.qws['w' + str(i)].tensor
                for i in range(len(self.learn.qws))]
        # w_grads = tf.gradients(tf.reduce_sum(
        #     tf.reduce_mean(self._log_py_xw, 1), 0), qws)
        n_batches = tf.cast(tf.shape(self.inputs)[0], tf.float32)
        w_grads = tf.gradients(self._log_py_xw / n_batches, qws)

        activations = [get_collection("a0"), get_collection("a1")]
        activations = [tf.concat(
            [activation,
             tf.ones(tf.concat([tf.shape(activation)[:-1], [1]], axis=0))], axis=-1) for activation in activations]

        s = [get_collection("s0"), get_collection("s1")]
        if self.stub == "regression":
            if self.config.true_fisher and self.stub == "regression":
                # True fisher: sample model and y from the var. distribution.
                sampled_log_prob = self.learn.sampled_log_prob
                s_grads = tf.gradients(tf.reduce_sum(sampled_log_prob), s)
            else:
                # Empirical fisher: sample model from var distribution, setting
                # y = target.
                s_grads = tf.gradients(self._log_py_xw, s)
        elif self.stub == "ird":
            assert self.config.true_fisher, "Only true fisher supported"
            # Sample model and y from the var. distribution.
            # (Yes, _log_py_xw holds true fisher even though in regression case
            # this is the empiral fisher).
            s_grads = [tf.check_numerics(x, "ird s_grads")
                for x in tf.gradients(self._log_py_xw, s)]

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
