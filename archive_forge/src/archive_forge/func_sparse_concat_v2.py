import numbers
import numpy as np
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_conversion
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import bincount_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_count_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import gen_sparse_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import special_math_ops
from tensorflow.python.ops.gen_sparse_ops import *
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.util import compat
from tensorflow.python.util import deprecation
from tensorflow.python.util import dispatch
from tensorflow.python.util import nest
from tensorflow.python.util import tf_inspect
from tensorflow.python.util.compat import collections_abc
from tensorflow.python.util.tf_export import get_canonical_name_for_symbol
from tensorflow.python.util.tf_export import tf_export
@tf_export('sparse.concat', v1=[])
def sparse_concat_v2(axis, sp_inputs, expand_nonconcat_dims=False, name=None):
    sp_inputs = _convert_to_sparse_tensors(sp_inputs)
    if len(sp_inputs) == 1:
        return sp_inputs[0]
    inds = [sp_input.indices for sp_input in sp_inputs]
    vals = [sp_input.values for sp_input in sp_inputs]
    shapes = [sp_input.dense_shape for sp_input in sp_inputs]
    if expand_nonconcat_dims:
        max_shape = math_ops.reduce_max(array_ops.concat([array_ops.reshape(shape, [1, -1]) for shape in shapes], 0), 0)
        shapes = [array_ops.concat([max_shape[:axis], shape[-1:] if axis == -1 else shape[axis:axis + 1], [] if axis == -1 else max_shape[axis + 1:]], 0) for shape in shapes]
    output_ind, output_val, output_shape = gen_sparse_ops.sparse_concat(inds, vals, shapes, axis, name=name)
    input_shapes = [inp.shape for inp in sp_inputs]
    if all((shape.rank is not None for shape in input_shapes)):
        if expand_nonconcat_dims:
            static_output_shape = []
            for dim in range(input_shapes[0].rank):
                static_output_shape.append(max((tensor_shape.dimension_at_index(shape, dim) for shape in input_shapes)))
        else:
            static_output_shape = input_shapes[0].as_list()
        static_output_shape[axis] = sum((tensor_shape.dimension_at_index(shape, axis) for shape in input_shapes))
    else:
        static_output_shape = tensor_shape.unknown_shape()
    if all((shape.is_fully_defined() for shape in input_shapes)):
        output_shape = ops.convert_to_tensor(static_output_shape, dtype=dtypes.int64)
        return sparse_tensor.SparseTensor(output_ind, output_val, output_shape)
    else:
        output = sparse_tensor.SparseTensor(output_ind, output_val, output_shape)
        output.set_shape(tensor_shape.TensorShape(static_output_shape))
        return output