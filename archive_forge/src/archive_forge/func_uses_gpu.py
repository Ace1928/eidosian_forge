import json
import shutil
import tempfile
import unittest
import numpy as np
from keras.src import backend
from keras.src import distribution
from keras.src import ops
from keras.src import utils
from keras.src.backend.common import is_float_dtype
from keras.src.backend.common import standardize_dtype
from keras.src.backend.common.global_state import clear_session
from keras.src.backend.common.keras_tensor import KerasTensor
from keras.src.models import Model
from keras.src.utils import traceback_utils
from keras.src.utils import tree
def uses_gpu():
    devices = distribution.list_devices()
    if any((d.startswith('gpu') for d in devices)):
        return True
    return False