import numpy as np
from tensorflow.python.framework import dtypes
from tensorflow.python.util import tf_export
Gets the default float type.

  Returns:
    If `is_prefer_float32()` is false and `is_allow_float64()` is true, returns
    float64; otherwise returns float32.
  