from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import inspect
import os

import numpy as np
import tensorflow as tf

from nng.misc.utils import get_logger, get_args, makedirs, NNG_DIR
from nng.misc.config import process_config
from nng.classification.misc.data_loader import load_pytorch
from nng.regression.misc.data_loader import generate_data_loader
from nng.classification.train import Trainer as ClassificationTrainer
from nng.regression.train import Trainer as RegressionTrainer
from nng.classification.model import Model as ClassificationModel
from nng.regression.model import Model as RegressionModel


_CLASSIFICATION_INPUT_DIM = {
    "cifar10": [32, 32, 3],
    "cifar100": [32, 32, 3],
}


def main():
    tf.set_random_seed(1231)
    np.random.seed(1231)

    config = None
    try:
        args = get_args()
        config = process_config(args.config)

        if config is None:
            raise Exception()
    except:
        print("Add a config file using \'--config file_name.json\'")
        exit(1)

    makedirs(config.summary_dir)
    makedirs(config.checkpoint_dir)

    # set logger
    path1 = os.path.join(NNG_DIR, 'classification/model.py')
    path2 = os.path.join(NNG_DIR, 'classification/train.py')
    path3 = os.path.join(NNG_DIR, 'regression/model.py')
    path4 = os.path.join(NNG_DIR, 'regression/train.py')
    logger = get_logger('log', logpath=config.summary_dir+'/',
                        filepath=os.path.abspath(__file__), package_files=[path1, path2, path3, path4])

    logger.info(config)

    # Define computational graph.
    sess = tf.Session()

    if config.mode == "classification":
        train_loader, test_loader = load_pytorch(config)

        model_ = ClassificationModel(config,
                                     _CLASSIFICATION_INPUT_DIM[config.dataset],
                                     len(train_loader.dataset))
        trainer = ClassificationTrainer(sess, model_, train_loader, test_loader, config, logger)

    elif config.mode == "regression":
        train_loader, test_loader, std_train, input_dim = generate_data_loader(config)
        config.std_train = std_train

        model_ = RegressionModel(config, input_dim, len(train_loader.dataset))
        trainer = RegressionTrainer(sess, model_, train_loader, test_loader, config, logger)

    else:
        print("Please choose either 'classification' or 'regression'.")
        raise NotImplementedError()

    trainer.train()


if __name__ == "__main__":
    main()
