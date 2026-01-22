import collections
import contextlib
import copy
import platform
import random
import threading
import numpy as np
import tensorflow.compat.v2 as tf
from absl import logging
from keras.src import backend
from keras.src.engine import keras_tensor
from keras.src.utils import object_identity
from keras.src.utils import tf_contextlib
from tensorflow.python.framework import ops
from tensorflow.python.util.tf_export import keras_export
from tensorflow.python import pywrap_tfe
def get_random_seed():
    """Retrieve a seed value to seed a random generator.

    Returns:
      the random seed as an integer.
    """
    if getattr(backend._SEED_GENERATOR, 'generator', None):
        return backend._SEED_GENERATOR.generator.randint(1, 1000000000.0)
    else:
        return random.randint(1, 1000000000.0)