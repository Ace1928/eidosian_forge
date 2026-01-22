import tensorflow.compat.v2 as tf
from keras.src.optimizers import adam as adam_experimental
from keras.src.optimizers.legacy import adadelta as adadelta_keras_v2
from keras.src.optimizers.legacy import adagrad as adagrad_keras_v2
from keras.src.optimizers.legacy import adam as adam_keras_v2
from keras.src.optimizers.legacy import adamax as adamax_keras_v2
from keras.src.optimizers.legacy import ftrl as ftrl_keras_v2
from keras.src.optimizers.legacy import (
from keras.src.optimizers.legacy import nadam as nadam_keras_v2
from keras.src.optimizers.legacy import rmsprop as rmsprop_keras_v2
def distributions_and_v1_and_v2_optimizers():
    """A common set of combination with DistributionStrategies and
    Optimizers."""
    return tf.__internal__.test.combinations.combine(distribution=[tf.__internal__.distribute.combinations.one_device_strategy, tf.__internal__.distribute.combinations.mirrored_strategy_with_gpu_and_cpu, tf.__internal__.distribute.combinations.mirrored_strategy_with_two_gpus, tf.__internal__.distribute.combinations.mirrored_strategy_with_two_gpus_no_merge_call], optimizer_fn=optimizers_v1_and_v2)