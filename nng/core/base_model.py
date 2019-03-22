from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import logging

import tensorflow as tf


class BaseModel:
    """ An abstract class for models. Children include models for classification and regression."""
    def __init__(self, config, logger=logging.getLogger("root")):
        self.config = config
        self.global_step_tensor = None
        self.init_global_step()
        self.logger = logger
        self.saver = None

    def save(self, sess):
        """ Saves the checkpoint in the path defined in the config file.
        :param sess: TensorFlow Session
        :return: None
        """
        self.logger.info("Saving model...")
        self.saver.save(sess, self.config.checkpoint_dir, self.global_step_tensor)
        self.logger.debug("Model saved")

    def load(self, sess):
        """ Loads latest checkpoint from the experiment path defined in the config file.
        :param sess: TensorFlow Session
        :return: None
        """
        latest_checkpoint = tf.train.latest_checkpoint(self.config.checkpoint_dir)
        if latest_checkpoint:
            self.logger.info(
                    "Loading model checkpoint {} ...\n".format(latest_checkpoint))
            self.saver.restore(sess, latest_checkpoint)
            self.logger.debug("Model loaded")

    def init_global_step(self):
        """ Initialize a TensorFlow variable to use it as a global step counter.
        :return: None
        """
        # Don't forget to add the global step tensor to the TensorFlow trainer.
        self.global_step_tensor = tf.train.get_or_create_global_step()

    def init_saver(self):
        """ Copy the following line in your child class.
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)
        """
        raise NotImplementedError()

    def build_model(self):
        raise NotImplementedError()
