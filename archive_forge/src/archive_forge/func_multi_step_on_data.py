import contextlib
import warnings
import numpy as np
import tensorflow as tf
import tree
from packaging.version import Version
from tensorflow.python.eager import context as tf_context
from keras.src import callbacks as callbacks_module
from keras.src import metrics as metrics_module
from keras.src import optimizers as optimizers_module
from keras.src.trainers import trainer as base_trainer
from keras.src.trainers.data_adapters import data_adapter_utils
from keras.src.trainers.epoch_iterator import EpochIterator
from keras.src.utils import traceback_utils
@tf.autograph.experimental.do_not_convert
def multi_step_on_data(data):
    outputs = one_step_on_data_distributed(data[:1])
    for single_step_data in data[1:]:
        step_outputs = one_step_on_data_distributed([single_step_data])
        outputs = tf.nest.map_structure(lambda t1, t2: concat([t1, t2]), outputs, step_outputs)
    return outputs