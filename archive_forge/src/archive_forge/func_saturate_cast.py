import builtins
import numbers
import numpy as np
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_conversion_registry
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import gen_bitwise_ops
from tensorflow.python.ops import gen_data_flow_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import gen_sparse_ops
from tensorflow.python.ops.gen_math_ops import *
from tensorflow.python.ops.numpy_ops import np_dtypes
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import compat
from tensorflow.python.util import deprecation
from tensorflow.python.util import dispatch
from tensorflow.python.util import nest
from tensorflow.python.util import tf_decorator
from tensorflow.python.util import traceback_utils
from tensorflow.python.util.compat import collections_abc
from tensorflow.python.util.tf_export import tf_export
@tf_export('dtypes.saturate_cast', 'saturate_cast')
@dispatch.register_unary_elementwise_api
@dispatch.add_dispatch_support
def saturate_cast(value, dtype, name=None):
    """Performs a safe saturating cast of `value` to `dtype`.

  This function casts the input to `dtype` without overflow.  If
  there is a danger that values would over or underflow in the cast, this op
  applies the appropriate clamping before the cast.  See `tf.cast` for more
  details.

  Args:
    value: A `Tensor`.
    dtype: The desired output `DType`.
    name: A name for the operation (optional).

  Returns:
    `value` safely cast to `dtype`.
  """
    with ops.name_scope(name, 'saturate_cast', [value]) as name:
        value = ops.convert_to_tensor(value, name='value')
        dtype = dtypes.as_dtype(dtype).base_dtype
        in_dtype = value.dtype
        if in_dtype.is_complex:
            if dtype.is_complex:
                real_in_dtype = in_dtype.real_dtype
                real_out_dtype = dtype.real_dtype
                if real_in_dtype.min < real_out_dtype.min or real_in_dtype.max > real_out_dtype.max:
                    value = gen_math_ops._clip_by_value(value, ops.convert_to_tensor(builtins.complex(real_out_dtype.min, real_out_dtype.min), dtype=in_dtype), ops.convert_to_tensor(builtins.complex(real_out_dtype.max, real_out_dtype.max), dtype=in_dtype), name='clamp')
                return cast(value, dtype, name=name)
            else:
                value = real(value)
                logging.warn('Casting complex to real discards imaginary part.')
                in_dtype = in_dtype.real_dtype
        out_real_dtype = dtype.real_dtype
        if in_dtype.min < out_real_dtype.min or in_dtype.max > out_real_dtype.max:
            np_dtype = in_dtype.as_numpy_dtype
            min_limit = np_dtype(np.maximum(in_dtype.min, out_real_dtype.min))
            if min_limit < out_real_dtype.min:
                min_limit = np.nextafter(min_limit, np_dtype(0), dtype=np_dtype)
            max_limit = np_dtype(np.minimum(in_dtype.max, out_real_dtype.max))
            if max_limit > out_real_dtype.max:
                max_limit = np.nextafter(max_limit, np_dtype(0), dtype=np_dtype)
            value = gen_math_ops._clip_by_value(value, ops.convert_to_tensor(min_limit, dtype=in_dtype), ops.convert_to_tensor(max_limit, dtype=in_dtype), name='clamp')
        return cast(value, dtype, name=name)