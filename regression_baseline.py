from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import os

from nng.misc.utils import get_logger, get_args, makedirs
from nng.misc.config import process_config
from nng.regression.misc.data_loader import generate_data_loader
from nng.regression.train import Trainer as RegressionTrainer
from nng.regression.model import Model as RegressionModel


def main():
    tf.set_random_seed(1231)
    np.random.seed(1231)
    try:
        args = get_args()
        config = process_config(args.config)
    except:
        print("Add a config file using \'--config file_name.json\'")
        exit(1)

    makedirs(config.summary_dir)
    makedirs(config.checkpoint_dir)

    # set logger
    logger = get_logger('log', logpath=config.summary_dir+'/',
                        filepath=os.path.abspath(__file__),
                        package_files=[], displaying=True)

    logger.info(config)
    if "expected_test_ll" in config:
        logger.info("Expected test log likelihood = {}".format(
            config.expected_test_ll))

    # Define computational graph
    rmse_results, ll_results = [], []
    n_runs = 10

    for i in range(1, n_runs + 1):
        sess = tf.Session()

        # Perform data splitting again with the provided seed.
        train_loader, test_loader, std_train, input_dim = \
            generate_data_loader(config, seed=i)
        config.std_train = std_train

        model_ = RegressionModel(config,
                                 input_dim,
                                 len(train_loader.dataset),
                                 logger=logger)
        trainer = RegressionTrainer(sess, model_, train_loader, test_loader, config, logger)

        trainer.train()

        rmse, ll = trainer.get_result()

        rmse_results.append(float(rmse))
        ll_results.append(float(ll))

        tf.reset_default_graph()

    for i, (rmse_result, ll_result) in enumerate(zip(rmse_results,
                                                     ll_results)):
        logger.info("\n## RUN {}".format(i))
        logger.info('# Test rmse = {}'.format(rmse_result))
        logger.info('# Test log likelihood = {}'.format(ll_result))

    logger.info("Results (mean/std. errors):")
    logger.info("Test rmse = {}/{}".format(
        np.mean(rmse_results), np.std(rmse_results) / n_runs ** 0.5))
    ll_mean, ll_std = np.mean(ll_results), np.std(ll_results)
    logger.info("Test log likelihood = {}/{}".format(
        np.mean(ll_results), np.std(ll_results) / n_runs ** 0.5))
    if "expected_test_ll" in config:
        logger.info("Expected test log likelihood = {}".format(
            config.expected_test_ll))
        if abs(config.expected_test_ll - ll_mean) > ll_std * 2:
            exit(1)
    exit(0)

if __name__ == "__main__":
    main()
