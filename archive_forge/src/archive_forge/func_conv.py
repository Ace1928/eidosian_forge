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
def conv(lhs, rhs, window_strides, padding, lhs_dilation, rhs_dilation, dimension_numbers, feature_group_count=1, precision_config=None, preferred_element_type=None, name=None, use_v2=False, batch_group_count=1):
    """Wraps the XLA ConvGeneralDilated operator.

  ConvGeneralDilated is the most general form of XLA convolution and is
  documented at
  https://www.tensorflow.org/performance/xla/operation_semantics#conv_convolution

  Args:
    lhs: the input tensor
    rhs: the kernel tensor
    window_strides: the inter-window strides
    padding: the padding to apply at the start and end of each input dimensions
    lhs_dilation: dilation to apply between input elements
    rhs_dilation: dilation to apply between kernel elements
    dimension_numbers: a `ConvolutionDimensionNumbers` proto.
    feature_group_count: number of feature groups for grouped convolution.
    precision_config: a `xla.PrecisionConfig` proto.
    preferred_element_type: the result `dtype`.
    name: an optional name for the operator.
    use_v2: an optional request to use the XlaConvV2 op even if not necessary.
    batch_group_count: number of batch groups or grouped filters.

  Returns:
    A tensor representing the output of the convolution.
  """
    precision_config_proto = ''
    if precision_config:
        precision_config_proto = precision_config.SerializeToString()
    needs_v2 = preferred_element_type or lhs.dtype != rhs.dtype or batch_group_count > 1
    if preferred_element_type is None:
        preferred_element_type = np_utils.result_type(lhs.dtype, rhs.dtype)
    if needs_v2 or use_v2:
        return gen_xla_ops.xla_conv_v2(lhs, rhs, window_strides=window_strides, padding=padding, lhs_dilation=lhs_dilation, rhs_dilation=rhs_dilation, feature_group_count=feature_group_count, batch_group_count=batch_group_count, dimension_numbers=dimension_numbers.SerializeToString(), precision_config=precision_config_proto, preferred_element_type=preferred_element_type, name=name)
    return gen_xla_ops.xla_conv(lhs, rhs, window_strides=window_strides, padding=padding, lhs_dilation=lhs_dilation, rhs_dilation=rhs_dilation, feature_group_count=feature_group_count, dimension_numbers=dimension_numbers.SerializeToString(), precision_config=precision_config_proto, name=name)