import functools
import operator
from tensorflow.python import pywrap_tfe
from tensorflow.python.eager import backprop_util
from tensorflow.python.eager import context
from tensorflow.python.eager import execute
from tensorflow.python.eager import imperative_grad
from tensorflow.python.eager import tape
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import composite_tensor_gradient
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import type_spec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_util
from tensorflow.python.ops import default_gradient
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops.parallel_for import control_flow_ops as pfor_ops
from tensorflow.python.ops.unconnected_gradients import UnconnectedGradients
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import _pywrap_utils
from tensorflow.python.util import nest
from tensorflow.python.util import tf_contextlib
from tensorflow.python.util import tf_inspect
from tensorflow.python.util import variable_utils
from tensorflow.python.util.tf_export import tf_export
def make_attr(attr_type, value):
    if attr_type == int(pywrap_tfe.TF_ATTR_TYPE):
        return dtypes.as_dtype(value)
    if attr_type == [int(pywrap_tfe.TF_ATTR_TYPE)]:
        return [dtypes.as_dtype(v) for v in value]
    if attr_type == int(pywrap_tfe.TF_ATTR_SHAPE):
        return tensor_shape.as_shape(value).as_proto()
    if attr_type == [int(pywrap_tfe.TF_ATTR_SHAPE)]:
        return [tensor_shape.as_shape(v).as_proto() for v in value]
    return nest.map_structure(lambda v: v.encode() if isinstance(v, str) else v, value)