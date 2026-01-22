import abc
import enum
import functools
import itertools
import os
from tensorflow.core.framework import variable_pb2
from tensorflow.python import pywrap_tensorflow  # pylint: disable=unused-import
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_conversion_registry
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.trackable import base as trackable
from tensorflow.python.util import _pywrap_utils
from tensorflow.python.util import object_identity
from tensorflow.python.util import tf_should_use
from tensorflow.python.util import traceback_utils
from tensorflow.python.util.deprecation import deprecated
from tensorflow.python.util.deprecation import deprecated_args
from tensorflow.python.util.tf_export import tf_export
@tf_export(v1=['assert_variables_initialized'])
@tf_should_use.should_use_result
def assert_variables_initialized(var_list=None):
    """Returns an Op to check if variables are initialized.

  NOTE: This function is obsolete and will be removed in 6 months.  Please
  change your implementation to use `report_uninitialized_variables()`.

  When run, the returned Op will raise the exception `FailedPreconditionError`
  if any of the variables has not yet been initialized.

  Note: This function is implemented by trying to fetch the values of the
  variables. If one of the variables is not initialized a message may be
  logged by the C++ runtime. This is expected.

  Args:
    var_list: List of `Variable` objects to check. Defaults to the value of
      `global_variables().`

  Returns:
    An Op, or None if there are no variables.
  """
    if var_list is None:
        var_list = global_variables() + local_variables()
    if not var_list:
        var_list = []
        for op in ops.get_default_graph().get_operations():
            if op.type in ['Variable', 'VariableV2', 'AutoReloadVariable']:
                var_list.append(op.outputs[0])
    if not var_list:
        return None
    else:
        ranks = []
        for var in var_list:
            with ops.colocate_with(var.op):
                ranks.append(array_ops.rank_internal(var, optimize=False))
        if len(ranks) == 1:
            return ranks[0]
        else:
            return array_ops_stack.stack(ranks)