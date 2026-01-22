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
def register_acd_resource_resolver(f):
    """Register a function for resolving resources touched by an op.

  `f` is called for every Operation added in the ACD context with the op's
  original resource reads and writes. `f` is expected to update the sets of
  resource reads and writes in-place and return True if it updated either of the
  sets, False otherwise.

  Example:
  @register_acd_resource_resolver
  def identity_resolver(op, resource_reads, resource_writes):
    # op: The `Operation` being processed by ACD currently.
    # resource_reads: An `ObjectIdentitySet` of read-only resources.
    # resource_writes: An `ObjectIdentitySet` of read-write resources.
    def update(resource_inputs):
      to_remove = []
      to_add = []
      for resource in resource_inputs:
        if resource.op.type == "Identity":
          to_remove.append(resource)
          to_add.extend(resource.op.inputs)
      for t in to_remove:
        resource_inputs.discard(t)
      resource_inputs.update(to_add)
      return to_add or to_remove
    return update(resource_reads) or update(resource_writes)

  Args:
    f: Python function with signature
    (Operation, ObjectIdentitySet, ObjectIdentitySet) -> bool

  Returns:
    The function `f` after adding it to the registry.
  """
    _acd_resource_resolvers_registry.register(f)
    return f