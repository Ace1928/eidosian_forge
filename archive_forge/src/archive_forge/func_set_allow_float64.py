import numpy as np
from tensorflow.python.framework import dtypes
from tensorflow.python.util import tf_export
def set_allow_float64(b):
    global _allow_float64
    _allow_float64 = b