from typing import Optional, Sequence, Union
import tensorflow.distribute.experimental.rpc.kernels.gen_rpc_ops as gen_rpc_ops
from tensorflow.distribute.experimental.rpc.proto import tf_rpc_service_pb2 as rpc_pb2
from tensorflow.python.data.util import structure
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.eager import function as tf_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import type_spec
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.saved_model import nested_structure_coder
from tensorflow.python.types import core as core_tf_types
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export
def call_blocking_wrapper(*args, timeout_in_ms=0):
    status_or, deleter = gen_rpc_ops.rpc_call(client_handle, args=validate_and_get_flat_inputs(*args), method_name=method_name, timeout_in_ms=timeout_in_ms)
    status_or = StatusOrResult(status_or, deleter, output_specs)
    if status_or.is_ok():
        return status_or.get_value()
    else:
        error_code, error_msg = status_or.get_error()
        raise errors.exception_type_from_error_code(error_code.numpy())(None, None, error_msg.numpy())