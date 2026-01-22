import platform
import warnings
import tensorflow.compat.v2 as tf
from absl import logging
from keras.src import backend
from keras.src.optimizers import adadelta
from keras.src.optimizers import adafactor
from keras.src.optimizers import adagrad
from keras.src.optimizers import adam
from keras.src.optimizers import adamax
from keras.src.optimizers import adamw
from keras.src.optimizers import ftrl
from keras.src.optimizers import lion
from keras.src.optimizers import nadam
from keras.src.optimizers import optimizer as base_optimizer
from keras.src.optimizers import rmsprop
from keras.src.optimizers import sgd
from keras.src.optimizers.legacy import adadelta as adadelta_legacy
from keras.src.optimizers.legacy import adagrad as adagrad_legacy
from keras.src.optimizers.legacy import adam as adam_legacy
from keras.src.optimizers.legacy import adamax as adamax_legacy
from keras.src.optimizers.legacy import ftrl as ftrl_legacy
from keras.src.optimizers.legacy import gradient_descent as gradient_descent_legacy
from keras.src.optimizers.legacy import nadam as nadam_legacy
from keras.src.optimizers.legacy import optimizer_v2 as base_optimizer_legacy
from keras.src.optimizers.legacy import rmsprop as rmsprop_legacy
from keras.src.optimizers.legacy.adadelta import Adadelta
from keras.src.optimizers.legacy.adagrad import Adagrad
from keras.src.optimizers.legacy.adam import Adam
from keras.src.optimizers.legacy.adamax import Adamax
from keras.src.optimizers.legacy.ftrl import Ftrl
from keras.src.optimizers.legacy.gradient_descent import SGD
from keras.src.optimizers.legacy.nadam import Nadam
from keras.src.optimizers.legacy.rmsprop import RMSprop
from keras.src.optimizers.optimizer_v1 import Optimizer
from keras.src.optimizers.optimizer_v1 import TFOptimizer
from keras.src.optimizers.schedules import learning_rate_schedule
from keras.src.saving.legacy import serialization as legacy_serialization
from keras.src.saving.serialization_lib import deserialize_keras_object
from keras.src.saving.serialization_lib import serialize_keras_object
from tensorflow.python.util.tf_export import keras_export
@keras_export('keras.__internal__.optimizers.convert_to_legacy_optimizer', v1=[])
def convert_to_legacy_optimizer(optimizer):
    """Convert experimental optimizer to legacy optimizer.

    This function takes in a `keras.optimizers.Optimizer`
    instance and converts it to the corresponding
    `keras.optimizers.legacy.Optimizer` instance.
    For example, `keras.optimizers.Adam(...)` to
    `keras.optimizers.legacy.Adam(...)`.

    Args:
        optimizer: An instance of `keras.optimizers.Optimizer`.
    """
    from keras.src.mixed_precision import loss_scale_optimizer
    if not isinstance(optimizer, base_optimizer.Optimizer):
        raise ValueError(f'`convert_to_legacy_optimizer` should only be called on instances of `tf.keras.optimizers.Optimizer`, but received {optimizer} of type {type(optimizer)}.')
    optimizer_name = optimizer.__class__.__name__.lower()
    config = optimizer.get_config()
    keys_to_remove = ['weight_decay', 'use_ema', 'ema_momentum', 'ema_overwrite_frequency', 'jit_compile', 'is_legacy_optimizer']
    for key in keys_to_remove:
        config.pop(key, None)
    if isinstance(optimizer, loss_scale_optimizer.LossScaleOptimizerV3):
        config['inner_optimizer'] = convert_to_legacy_optimizer(optimizer.inner_optimizer)
        if optimizer_name == 'lossscaleoptimizerv3':
            optimizer_name = 'lossscaleoptimizer'
    if hasattr(optimizer, '_learning_rate') and isinstance(optimizer._learning_rate, learning_rate_schedule.LearningRateSchedule):
        config['learning_rate'] = optimizer._learning_rate
    legacy_optimizer_config = {'class_name': optimizer_name, 'config': config}
    return deserialize(legacy_optimizer_config, use_legacy_optimizer=True)