from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import lookup_ops
from tensorflow.python.ops import variables
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export
@tf_export(v1=['saved_model.main_op_with_restore', 'saved_model.main_op.main_op_with_restore'])
@deprecation.deprecated(None, _DEPRECATION_MSG)
def main_op_with_restore(restore_op_name):
    """Returns a main op to init variables, tables and restore the graph.

  Returns the main op including the group of ops that initializes all
  variables, initialize local variables, initialize all tables and the restore
  op name.

  Args:
    restore_op_name: Name of the op to use to restore the graph.

  Returns:
    The set of ops to be run as part of the main op upon the load operation.
  """
    with ops.control_dependencies([main_op()]):
        main_op_with_restore = control_flow_ops.group(restore_op_name)
    return main_op_with_restore