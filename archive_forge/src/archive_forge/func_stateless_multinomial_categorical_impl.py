import numpy as np
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import gen_random_index_shuffle_ops
from tensorflow.python.ops import gen_stateless_random_ops
from tensorflow.python.ops import gen_stateless_random_ops_v2
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops_util
from tensorflow.python.ops import shape_util
from tensorflow.python.util import deprecation
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export
def stateless_multinomial_categorical_impl(logits, num_samples, dtype, seed):
    """Implementation for stateless multinomial/categorical ops (v1/v2)."""
    logits = ops.convert_to_tensor(logits, name='logits')
    dtype = dtypes.as_dtype(dtype) if dtype else dtypes.int64
    accepted_dtypes = (dtypes.int32, dtypes.int64)
    if dtype not in accepted_dtypes:
        raise ValueError(f'Argument `dtype` got invalid value {dtype}. Accepted dtypes are {accepted_dtypes}.')
    return gen_stateless_random_ops.stateless_multinomial(logits, num_samples, seed, output_dtype=dtype)