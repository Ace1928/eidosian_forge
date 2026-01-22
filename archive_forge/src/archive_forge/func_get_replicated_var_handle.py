from typing import Any, Callable, List, Optional, Text, Tuple, Union
from absl import logging
from tensorflow.core.framework import attr_value_pb2
from tensorflow.python.distribute import device_util
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.framework import device as pydev
from tensorflow.python.framework import errors
from tensorflow.python.framework import func_graph
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import variables
from tensorflow.python.tpu import device_assignment as device_assignment_lib
from tensorflow.python.tpu.ops import tpu_ops
from tensorflow.python.types import core as core_types
from tensorflow.python.util import compat
from tensorflow.python.util.tf_export import tf_export
def get_replicated_var_handle(self, name: Text, handle_id: Text, vars_: Union[List[core_types.Tensor], List[variables.Variable]], is_mirrored: bool=False, is_packed: bool=False) -> core_types.Tensor:
    """Returns a variable handle for replicated TPU variable 'var'.

    This is a method used by an experimental replicated variable implementation
    and is not intended as a public API.

    Args:
      name: The common name of the variable.
      handle_id: Unique ID of the variable handle, used as the cache key.
      vars_: The replicated TPU variables or handles.
      is_mirrored: Whether the variables are mirrored, which guarantees the
        values in each replica are always the same.
      is_packed: Whether the replicated variables are packed into one variable.

    Returns:
      The handle of the TPU replicated input node.
    """
    device_assignment = _enclosing_tpu_device_assignment()
    handle = self._replicated_vars.get(handle_id)
    if handle is not None:
        return handle
    if device_assignment is not None and (not is_packed):
        job_name = pydev.DeviceSpec.from_string(vars_[0].device).job
        devices_to_vars = {device_util.canonicalize(v.device): v for v in vars_}
        replicated_vars = []
        for replica_id in range(device_assignment.num_replicas):
            for logical_core in range(device_assignment.num_cores_per_replica):
                device = device_util.canonicalize(device_assignment.tpu_device(replica=replica_id, logical_core=logical_core, job=job_name))
                if device in devices_to_vars:
                    replicated_vars.append(devices_to_vars[device])
                    break
            else:
                raise ValueError('Failed to find a variable on any device in replica {} for current device assignment'.format(replica_id))
    else:
        replicated_vars = vars_
    _, graph = _enclosing_tpu_context_and_graph()
    with graph.as_default():
        if isinstance(replicated_vars[0], variables.Variable):
            replicated_vars = [v.handle for v in replicated_vars]
        saved_context = graph._get_control_flow_context()
        graph._set_control_flow_context(self.outer_context)
        handle = tpu_ops.tpu_replicated_input(replicated_vars, name=name + '/handle', is_mirrored_variable=is_mirrored, is_packed=is_packed)
        graph._set_control_flow_context(saved_context)
    self._replicated_vars[handle_id] = handle
    return handle