import builtins
import enum
import functools
import math
import numbers
import numpy as np
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import control_flow_assert
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import manip_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import sort_ops
from tensorflow.python.ops.numpy_ops import np_arrays
from tensorflow.python.ops.numpy_ops import np_dtypes
from tensorflow.python.ops.numpy_ops import np_utils
from tensorflow.python.util import nest
from tensorflow.python.util import tf_export
@tf_export.tf_export('experimental.numpy.moveaxis', v1=[])
@np_utils.np_doc('moveaxis')
def moveaxis(a, source, destination):
    """Raises ValueError if source, destination not in (-ndim(a), ndim(a))."""
    if not source and (not destination):
        return a
    a = asarray(a)
    if isinstance(source, int):
        source = (source,)
    if isinstance(destination, int):
        destination = (destination,)
    if len(source) != len(destination):
        raise ValueError('The lengths of source and destination must equal')
    a_rank = np_utils._maybe_static(array_ops.rank(a))

    def _correct_axis(axis, rank):
        if axis < 0:
            return axis + rank
        return axis
    source = tuple((_correct_axis(axis, a_rank) for axis in source))
    destination = tuple((_correct_axis(axis, a_rank) for axis in destination))
    if a.shape.rank is not None:
        perm = [i for i in range(a_rank) if i not in source]
        for dest, src in sorted(zip(destination, source)):
            assert dest <= len(perm)
            perm.insert(dest, src)
    else:
        r = math_ops.range(a_rank)

        def _remove_indices(a, b):
            """Remove indices (`b`) from `a`."""
            items = array_ops_stack.unstack(sort_ops.sort(array_ops_stack.stack(b)), num=len(b))
            i = 0
            result = []
            for item in items:
                result.append(a[i:item])
                i = item + 1
            result.append(a[i:])
            return array_ops.concat(result, 0)
        minus_sources = _remove_indices(r, source)
        minus_dest = _remove_indices(r, destination)
        perm = array_ops.scatter_nd(array_ops.expand_dims(minus_dest, 1), minus_sources, [a_rank])
        perm = array_ops.tensor_scatter_update(perm, array_ops.expand_dims(destination, 1), source)
    a = array_ops.transpose(a, perm)
    return a