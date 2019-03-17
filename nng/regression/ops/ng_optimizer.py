from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from abc import ABC, abstractmethod

import tensorflow as tf


class NGOptimizer(ABC):
    def __init__(self, shape, N, lam, alpha, beta, w_name):
        self.shape = shape
        self.N = N
        self.lam = lam
        self.alpha = alpha
        self.beta = beta
        self.w_name = w_name

    @abstractmethod
    def update(self, w, w_grad, a, s_grad):
        pass

    @abstractmethod
    def push_collection(self, add_summary=True):
        pass
