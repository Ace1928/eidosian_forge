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
def strategy_minus_tpu_and_input_config_combinations_eager():
    return tf.__internal__.test.combinations.times(tf.__internal__.test.combinations.combine(distribution=strategies_minus_tpu), eager_mode_test_configuration())