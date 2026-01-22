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
@tf_export.tf_export('experimental.numpy.result_type', v1=[])
@np_doc_only('result_type')
def result_type(*arrays_and_dtypes):
    if ops.is_auto_dtype_conversion_enabled():
        dtype, _ = flexible_dtypes.result_type(*arrays_and_dtypes)
        return dtype
    arrays_and_dtypes = [_maybe_get_dtype(x) for x in nest.flatten(arrays_and_dtypes)]
    if not arrays_and_dtypes:
        arrays_and_dtypes = [np.asarray([])]
    return np_dtypes._result_type(*arrays_and_dtypes)