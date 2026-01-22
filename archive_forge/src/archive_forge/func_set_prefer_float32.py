import numpy as np
from tensorflow.python.framework import dtypes
from tensorflow.python.util import tf_export
def set_prefer_float32(b):
    global _prefer_float32
    _prefer_float32 = b