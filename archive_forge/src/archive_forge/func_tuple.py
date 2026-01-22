import abc
from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.protobuf import control_flow_pb2
from tensorflow.python.eager import context
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import type_spec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_util as util
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import gen_control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops.gen_control_flow_ops import *
from tensorflow.python.util import compat
from tensorflow.python.util import dispatch
from tensorflow.python.util import nest
from tensorflow.python.util import variable_utils
from tensorflow.python.util.tf_export import tf_export
@tf_export(v1=['tuple'])
@dispatch.add_dispatch_support
def tuple(tensors, name=None, control_inputs=None):
    """Group tensors together.

  This creates a tuple of tensors with the same values as the `tensors`
  argument, except that the value of each tensor is only returned after the
  values of all tensors have been computed.

  `control_inputs` contains additional ops that have to finish before this op
  finishes, but whose outputs are not returned.

  This can be used as a "join" mechanism for parallel computations: all the
  argument tensors can be computed in parallel, but the values of any tensor
  returned by `tuple` are only available after all the parallel computations
  are done.

  See also `tf.group` and
  `tf.control_dependencies`.

  Args:
    tensors: A list of `Tensor`s or `IndexedSlices`, some entries can be `None`.
    name: (optional) A name to use as a `name_scope` for the operation.
    control_inputs: List of additional ops to finish before returning.

  Returns:
    Same as `tensors`.

  Raises:
    ValueError: If `tensors` does not contain any `Tensor` or `IndexedSlices`.
    TypeError: If `control_inputs` is not a list of `Operation` or `Tensor`
      objects.

  """
    if context.executing_eagerly():
        return tensors
    with ops.name_scope(name, 'tuple', tensors) as name:
        tensors = [t if isinstance(t, ops.Operation) or tensor_util.is_tf_type(t) or t is None else ops.convert_to_tensor(t) for t in tensors]
        gating_ops = [t if isinstance(t, ops.Operation) else t.op for t in tensors if t is not None]
        if control_inputs:
            for c in control_inputs:
                if isinstance(c, tensor_lib.Tensor):
                    c = c.op
                elif not isinstance(c, ops.Operation):
                    raise TypeError(f"'control_inputs' must only contain Operation or Tensor. Received: {type(c)}")
                gating_ops.append(c)
        gating_ops = sorted(set(gating_ops), key=lambda op: op._id)
        if not gating_ops:
            raise ValueError(f"'tensors' must have at least one Tensor. Received: {tensors}.")
        gate = group(*gating_ops)
        tpl = []
        for t in tensors:
            if tensor_util.is_tf_type(t):
                tpl.append(with_dependencies([gate], t))
            elif isinstance(t, ops.Operation):
                with ops.control_dependencies([gate]):
                    tpl.append(group(t))
            else:
                tpl.append(None)
        return tpl