from unittest import mock
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers.experimental import preprocessing
import autokeras as ak
from autokeras import keras_layers
from autokeras import test_utils
from autokeras.engine import tuner as tuner_module
from autokeras.tuners import greedy
def called_with_early_stopping(func):
    callbacks = func.call_args_list[0][1]['callbacks']
    return any([isinstance(callback, keras.callbacks.EarlyStopping) for callback in callbacks])