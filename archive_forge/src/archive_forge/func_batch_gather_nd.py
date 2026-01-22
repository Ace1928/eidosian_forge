import numbers
import numpy as np
from tensorflow.core.config import flags
from tensorflow.python.eager import context
from tensorflow.python.eager import record
from tensorflow.python.framework import common_shapes
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_conversion_registry
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework.constant_op import constant
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import shape_util
from tensorflow.python.ops.gen_array_ops import *
from tensorflow.python.ops.gen_array_ops import reverse_v2 as reverse  # pylint: disable=unused-import
from tensorflow.python.types import core
from tensorflow.python.util import _pywrap_utils
from tensorflow.python.util import deprecation
from tensorflow.python.util import dispatch
from tensorflow.python.util import nest
from tensorflow.python.util import tf_decorator
from tensorflow.python.util.lazy_loader import LazyLoader
from tensorflow.python.util.tf_export import tf_export
def batch_gather_nd(params, indices, batch_dims, name=None):
    """gather_nd implementation with batch support."""
    with ops.name_scope(name, 'BatchGatherND', [params, indices]):
        indices = ops.convert_to_tensor(indices, name='indices')
        params = ops.convert_to_tensor(params, name='params')
        if not isinstance(batch_dims, int):
            raise TypeError(f'Argument `batch_dims` must be an int; got {batch_dims}')
        if batch_dims < 0:
            raise ValueError('tf.gather_nd does not allow negative batch_dims.')
        params_ndims = params.shape.ndims
        indices_ndims = indices.shape.ndims
        if indices_ndims is not None and batch_dims >= indices_ndims:
            raise ValueError(f'Argument `batch_dims` = {batch_dims} must be less than rank(`indices`) = {indices_ndims}')
        if params_ndims is not None and batch_dims >= params_ndims:
            raise ValueError(f'Argument `batch_dims` = {batch_dims} must be less than rank(`params`) = {params_ndims}')
        expand = batch_dims == 0
        if expand:
            params = expand_dims(params, axis=0)
            indices = expand_dims(indices, axis=0)
            batch_dims = 1
        params_shape = shape(params)
        indices_shape = shape(indices)
        batch_shape = params_shape[:batch_dims]
        batch_size = gen_math_ops.prod(batch_shape, [0])
        index_internal_ndims = rank(indices) - batch_dims - 1
        indices_internal_shape = indices_shape[batch_dims:-1]
        batch_dim_list = array_ops_stack.unstack(batch_shape, axis=0)
        dim_ranges = [gen_math_ops.cast(gen_math_ops._range(0, gen_math_ops.cast(x, dtypes.int32), 1), indices.dtype) for x in batch_dim_list]
        mesh_list = meshgrid(*dim_ranges, indexing='ij') if dim_ranges else []
        flat_list = [reshape(x, shape=(-1,)) for x in mesh_list]
        index_grid = transpose(array_ops_stack.stack(flat_list, axis=0))
        index_grid_shape = shape(index_grid)
        index_grid = reshape(index_grid, concat([index_grid_shape[:1], ones(index_internal_ndims, dtype=index_grid_shape.dtype), index_grid_shape[1:]], axis=0))
        tile_shape = concat(((1,), indices_internal_shape, (1,)), axis=0)
        index_grid = tile(index_grid, multiples=tile_shape)
        flat_shape = concat(([batch_size], indices_shape[batch_dims:]), axis=0)
        flat_indices = reshape(indices, shape=flat_shape)
        indices = concat((index_grid, flat_indices), axis=-1)
        out = gen_array_ops.gather_nd(params, indices)
        out_shape = shape(out)
        out = reshape(out, shape=concat((batch_shape, out_shape[1:]), axis=0))
        if expand:
            out = squeeze(out, axis=0)
    return out