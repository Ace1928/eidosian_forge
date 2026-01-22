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
def can_jit_compile(warn=False):
    """Returns True if TensorFlow XLA is available for the platform."""
    if platform.system() == 'Darwin' and 'arm' in platform.processor().lower():
        if warn:
            logging.warning('XLA (`jit_compile`) is not yet supported on Apple M1/M2 ARM processors. Falling back to `jit_compile=False`.')
        return False
    if pywrap_tfe.TF_ListPluggablePhysicalDevices():
        if warn:
            logging.warning('XLA (`jit_compile`) is not supported on your system. Falling back to `jit_compile=False`.')
        return False
    return True