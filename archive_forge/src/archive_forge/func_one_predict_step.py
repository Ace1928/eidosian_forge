import numpy as np
import tree
from keras.src import backend
from keras.src import callbacks as callbacks_module
from keras.src.backend.common import standardize_dtype
from keras.src.backend.common.keras_tensor import KerasTensor
from keras.src.backend.numpy.core import is_tensor
from keras.src.trainers import trainer as base_trainer
from keras.src.trainers.data_adapters import data_adapter_utils
from keras.src.trainers.epoch_iterator import EpochIterator
from keras.src.utils import traceback_utils
def one_predict_step(data):
    data = data[0]
    return self.predict_step(data)