import random as python_random
import numpy as np
from keras.src import backend
from keras.src.api_export import keras_export
from keras.src.backend.common import global_state
from keras.src.utils import jax_utils
def make_default_seed():
    return python_random.randint(1, int(1000000000.0))