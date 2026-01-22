from tensorflow.compiler.tf2xla.ops import gen_xla_ops
from tensorflow.compiler.xla import xla_data_pb2
from tensorflow.core.framework import attr_value_pb2
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import bitwise_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import gen_random_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import random_ops_util
from tensorflow.python.ops import special_math_ops
from tensorflow.python.ops.numpy_ops import np_utils
def dot_general(lhs, rhs, dimension_numbers, precision_config=None, preferred_element_type=None, name=None, use_v2=False):
    precision_config_proto = ''
    if precision_config:
        precision_config_proto = precision_config.SerializeToString()
    needs_v2 = preferred_element_type or lhs.dtype != rhs.dtype
    if preferred_element_type is None:
        preferred_element_type = np_utils.result_type(lhs.dtype, rhs.dtype)
    if needs_v2 or use_v2:
        return gen_xla_ops.xla_dot_v2(lhs, rhs, dimension_numbers=dimension_numbers.SerializeToString(), precision_config=precision_config_proto, preferred_element_type=preferred_element_type, name=name)
    return gen_xla_ops.xla_dot(lhs, rhs, dimension_numbers=dimension_numbers.SerializeToString(), precision_config=precision_config_proto, name=name)