import collections
import enum
from tensorflow.core.framework import attr_value_pb2
from tensorflow.python.eager import context
from tensorflow.python.framework import auto_control_deps_utils as utils
from tensorflow.python.framework import dtypes as dtypes_module
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import op_def_registry
from tensorflow.python.framework import ops
from tensorflow.python.framework import registry
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import control_flow_util
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.util import nest
from tensorflow.python.util import object_identity
from tensorflow.python.util import tf_decorator
def op_is_stateful(op):
    ret = op._is_stateful and (op.type not in ASYNC_STATEFUL_OPS and op.type not in LEGACY_RANDOM_OPS and (op.type not in SKIPPED_ORDER_INSENSITIVE_STATEFUL_OPS)) or op.type in _ALLOWLIST_STATELESS_OPS
    return ret