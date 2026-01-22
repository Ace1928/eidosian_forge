import inspect
import numbers
import os
import re
import numpy as np
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import flexible_dtypes
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import cond as tf_cond
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.numpy_ops import np_arrays
from tensorflow.python.ops.numpy_ops import np_dtypes
from tensorflow.python.types import core
from tensorflow.python.util import nest
from tensorflow.python.util import tf_export
def set_np_doc_form(value):
    """Selects the form of the original numpy docstrings.

  This function sets a global variable that controls how a tf-numpy symbol's
  docstring should refer to the original numpy docstring. If `value` is
  `'inlined'`, the numpy docstring will be verbatim copied into the tf-numpy
  docstring. Otherwise, a link to the original numpy docstring will be
  added. Which numpy version the link points to depends on `value`:
  * `'stable'`: the current stable version;
  * `'dev'`: the current development version;
  * pattern `\\d+(\\.\\d+(\\.\\d+)?)?`: `value` will be treated as a version number,
    e.g. '1.16'.

  Args:
    value: the value to set the global variable to.
  """
    global _np_doc_form
    _np_doc_form = value