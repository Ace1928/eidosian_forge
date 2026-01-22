import functools
import numpy as np
import tensorflow.compat.v2 as tf
from absl.testing import parameterized
import keras.src as keras
from keras.src.distribute import distributed_training_utils
from keras.src.distribute.strategy_combinations import all_strategies
from keras.src.distribute.strategy_combinations import (
from keras.src.distribute.strategy_combinations import strategies_minus_tpu
from keras.src.mixed_precision import policy
from keras.src.utils import data_utils
def strategies_for_embedding_models():
    """Returns distribution strategies to test for embedding models.

    Since embedding models take longer to train, we disregard DefaultStrategy
    in order to prevent testing timeouts.
    """
    return [s for s in all_strategies if s.required_tpu or s.required_gpus or s is tf.__internal__.distribute.combinations.one_device_strategy]