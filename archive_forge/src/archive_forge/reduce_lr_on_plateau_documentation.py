import warnings
import numpy as np
from keras.src import backend
from keras.src.api_export import keras_export
from keras.src.callbacks.callback import Callback
from keras.src.utils import io_utils
Resets wait counter and cooldown counter.