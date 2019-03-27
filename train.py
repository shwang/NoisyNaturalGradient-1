import os

import numpy as np
import tensorflow as tf

from nng.misc.utils import get_logger, get_args, makedirs, NNG_DIR
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
    paths = [os.path.join(NNG_DIR, 'regression/model.py'),
             os.path.join(NNG_DIR, 'regression/train.py')]
    logger = get_logger('log', logpath=config.summary_dir+'/',
                        filepath=os.path.abspath(__file__),
                        package_files=paths,
                        displaying=True)



    logger.info(config)

    sess = tf.Session()

    if config.mode == "classification":
        raise NotImplementedError
    elif config.mode == "regression":
        train_loader, test_loader, std_train, input_dim = generate_data_loader(config)
        config.std_train = std_train

        model_ = RegressionModel(config, input_dim, len(train_loader.dataset),
                                 logger=logger)
        trainer = RegressionTrainer(sess, model_, train_loader, test_loader, config, logger)  # pytype: disable=wrong-arg-types

    else:
        raise ValueError(config.mode)

    trainer.train()


if __name__ == "__main__":
    main()
