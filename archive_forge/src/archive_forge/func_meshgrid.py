import numbers
import sys
import numpy as np
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import bitwise_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import control_flow_assert
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import sort_ops
from tensorflow.python.ops import special_math_ops
from tensorflow.python.ops import while_loop
from tensorflow.python.ops.numpy_ops import np_array_ops
from tensorflow.python.ops.numpy_ops import np_arrays
from tensorflow.python.ops.numpy_ops import np_dtypes
from tensorflow.python.ops.numpy_ops import np_utils
from tensorflow.python.util import tf_export
@tf_export.tf_export('experimental.numpy.meshgrid', v1=[])
@np_utils.np_doc('meshgrid')
def meshgrid(*xi, **kwargs):
    """This currently requires copy=True and sparse=False."""
    sparse = kwargs.get('sparse', False)
    if sparse:
        raise ValueError(f'Function `meshgrid` does not support returning sparse arrays yet. Received: sparse={sparse}')
    copy = kwargs.get('copy', True)
    if not copy:
        raise ValueError(f'Function `meshgrid` only supports copy=True. Received: copy={copy}')
    indexing = kwargs.get('indexing', 'xy')
    xi = [np_array_ops.asarray(arg) for arg in xi]
    kwargs = {'indexing': indexing}
    outputs = array_ops.meshgrid(*xi, **kwargs)
    return outputs