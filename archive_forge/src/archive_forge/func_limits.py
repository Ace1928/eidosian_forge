import abc
import builtins
import dataclasses
from typing import Type, Sequence, Optional
import numpy as np
from tensorflow.core.framework import types_pb2
from tensorflow.python import pywrap_tensorflow  # pylint: disable=unused-import
from tensorflow.python.framework import _dtypes
from tensorflow.python.framework import cpp_shape_inference_pb2
from tensorflow.python.types import doc_typealias
from tensorflow.python.util.tf_export import tf_export
from tensorflow.python.types import trace
from tensorflow.core.function import trace_type
from tensorflow.tools.docs import doc_controls
from tensorflow.tsl.python.lib.core import pywrap_ml_dtypes
@property
def limits(self, clip_negative=True):
    """Return intensity limits, i.e.

    (min, max) tuple, of the dtype.
    Args:
      clip_negative : bool, optional If True, clip the negative range (i.e.
        return 0 for min intensity) even if the image dtype allows negative
        values. Returns
      min, max : tuple Lower and upper intensity limits.
    """
    if self.as_numpy_dtype in dtype_range:
        min, max = dtype_range[self.as_numpy_dtype]
    else:
        raise ValueError(str(self) + ' does not have defined limits.')
    if clip_negative:
        min = 0
    return (min, max)