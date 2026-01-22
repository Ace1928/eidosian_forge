import collections
import contextlib
import copy
import enum  # pylint: disable=g-bad-import-order
import functools
import threading
import weakref
import six
from tensorflow.python import tf2
from tensorflow.python.autograph.core import ag_ctx as autograph_ctx
from tensorflow.python.autograph.impl import api as autograph
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.distribute import collective_util
from tensorflow.python.distribute import device_util
from tensorflow.python.distribute import numpy_dataset
from tensorflow.python.distribute import reduce_util
from tensorflow.python.eager import context as eager_context
from tensorflow.python.eager import def_function
from tensorflow.python.eager import monitoring
from tensorflow.python.eager import tape
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import custom_gradient
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import ref_variable
from tensorflow.python.ops import summary_ops_v2
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variable_v1
from tensorflow.python.platform import tf_logging
from tensorflow.python.trackable import base as trackable
from tensorflow.python.types import distribute as ds_types
from tensorflow.python.util import deprecation
from tensorflow.python.util import nest
from tensorflow.python.util import tf_contextlib
from tensorflow.python.util.deprecation import deprecated
from tensorflow.python.util.tf_export import tf_export
from tensorflow.tools.docs import doc_controls
def mean_reduce_helper(v, axes=axis):
    """Computes the numerator and denominator on each replica."""
    numer = math_ops.reduce_sum(v, axis=axes)

    def dimension(axis):
        if v.shape.rank is not None:
            if axis < 0:
                if axis + v.shape.rank < 0:
                    raise ValueError('`axis` = %r out of range for `value` with rank %d' % (axis, v.shape.rank))
                axis += v.shape.rank
            elif axis >= v.shape.rank:
                raise ValueError('`axis` = %r out of range for `value` with rank %d' % (axis, v.shape.rank))
            dim = tensor_shape.dimension_value(v.shape[axis])
            if dim is not None:
                return array_ops.identity(constant_op.constant(dim, dtype=dtypes.int64))
        elif axis < 0:
            axis = axis + array_ops.rank(v)
        return array_ops.identity(array_ops.shape_v2(v, out_type=dtypes.int64)[axis])
    if isinstance(axis, six.integer_types):
        denom = dimension(axis)
    elif isinstance(axis, (tuple, list)):
        denom = math_ops.reduce_prod([dimension(a) for a in axes])
    else:
        raise TypeError('Expected `axis` to be an integer, tuple or list not: %r' % axis)
    return (numer, denom)