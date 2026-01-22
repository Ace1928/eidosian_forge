import numpy as np
from tensorflow.python.framework import dtypes
from tensorflow.python.util import tf_export
def canonicalize_dtype(dtype):
    if not _allow_float64:
        try:
            return _to_float32[dtype]
        except KeyError:
            pass
    return dtype