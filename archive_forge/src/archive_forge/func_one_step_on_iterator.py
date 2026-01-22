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
def one_step_on_iterator(iterator):
    """Runs a single test step given a Dataset iterator."""
    data = next(iterator)
    outputs = self.distribute_strategy.run(one_step_on_data, args=(data,))
    outputs = reduce_per_replica(outputs, self.distribute_strategy, reduction=self.distribute_reduction_method)
    return outputs