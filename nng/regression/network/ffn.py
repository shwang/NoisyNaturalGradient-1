from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf

from nng.misc.registry import register_model
from nng.regression.misc.collections import add_to_collection
from nng.regression.ops import emvg_optimizer, mvg_optimizer


def ffn(layer_type, input_size, num_data, kl_factor, ita, alpha, beta, damp,
        *, layer_sizes, omega=None):
    valid_layer_type = ["emvg", "mvg"]
    layer_type = layer_type.lower()
    if layer_type not in valid_layer_type:
        raise ValueError("Unavailable layer type %s" % layer_type)

    init_ops = []
    layers = []

    if layer_type == "emvg":
        optim_cls = emvg_optimizer.EMVGOptimizer
    elif layer_type == "mvg":
        optim_cls = mvg_optimizer.MVGOptimizer

    for i, (n_in, n_out) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        kwargs = dict(omega=omega) if layer_type == "emvg" else {}
        layer = optim_cls([n_in + 1, n_out], num_data, kl_factor, ita,
                alpha, beta, damp, "w{}".format(i), **kwargs)
        layers.append(layer)
        layer.push_collection()
        if layer_type == "emvg":
            init_ops.append(layer.init_r_kfac())  # pytype: disable=attribute-error

    return layers, init_ops
