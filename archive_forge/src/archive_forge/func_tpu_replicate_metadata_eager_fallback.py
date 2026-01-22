import collections
from tensorflow.python import pywrap_tfe as pywrap_tfe
from tensorflow.python.eager import context as _context
from tensorflow.python.eager import core as _core
from tensorflow.python.eager import execute as _execute
from tensorflow.python.framework import dtypes as _dtypes
from tensorflow.security.fuzzing.py import annotation_types as _atypes
from tensorflow.python.framework import op_def_registry as _op_def_registry
from tensorflow.python.framework import ops as _ops
from tensorflow.python.framework import op_def_library as _op_def_library
from tensorflow.python.util.deprecation import deprecated_endpoints
from tensorflow.python.util import dispatch as _dispatch
from tensorflow.python.util.tf_export import tf_export
from typing import TypeVar, List
def tpu_replicate_metadata_eager_fallback(num_replicas: int, num_cores_per_replica: int, topology: str, use_tpu: bool, device_assignment, computation_shape, host_compute_core, padding_map, step_marker_location: str, allow_soft_placement: bool, use_spmd_for_xla_partitioning: bool, tpu_compile_options_proto: str, name, ctx):
    num_replicas = _execute.make_int(num_replicas, 'num_replicas')
    if num_cores_per_replica is None:
        num_cores_per_replica = 1
    num_cores_per_replica = _execute.make_int(num_cores_per_replica, 'num_cores_per_replica')
    if topology is None:
        topology = ''
    topology = _execute.make_str(topology, 'topology')
    if use_tpu is None:
        use_tpu = True
    use_tpu = _execute.make_bool(use_tpu, 'use_tpu')
    if device_assignment is None:
        device_assignment = []
    if not isinstance(device_assignment, (list, tuple)):
        raise TypeError("Expected list for 'device_assignment' argument to 'tpu_replicate_metadata' Op, not %r." % device_assignment)
    device_assignment = [_execute.make_int(_i, 'device_assignment') for _i in device_assignment]
    if computation_shape is None:
        computation_shape = []
    if not isinstance(computation_shape, (list, tuple)):
        raise TypeError("Expected list for 'computation_shape' argument to 'tpu_replicate_metadata' Op, not %r." % computation_shape)
    computation_shape = [_execute.make_int(_i, 'computation_shape') for _i in computation_shape]
    if host_compute_core is None:
        host_compute_core = []
    if not isinstance(host_compute_core, (list, tuple)):
        raise TypeError("Expected list for 'host_compute_core' argument to 'tpu_replicate_metadata' Op, not %r." % host_compute_core)
    host_compute_core = [_execute.make_str(_s, 'host_compute_core') for _s in host_compute_core]
    if padding_map is None:
        padding_map = []
    if not isinstance(padding_map, (list, tuple)):
        raise TypeError("Expected list for 'padding_map' argument to 'tpu_replicate_metadata' Op, not %r." % padding_map)
    padding_map = [_execute.make_str(_s, 'padding_map') for _s in padding_map]
    if step_marker_location is None:
        step_marker_location = 'STEP_MARK_AT_ENTRY'
    step_marker_location = _execute.make_str(step_marker_location, 'step_marker_location')
    if allow_soft_placement is None:
        allow_soft_placement = False
    allow_soft_placement = _execute.make_bool(allow_soft_placement, 'allow_soft_placement')
    if use_spmd_for_xla_partitioning is None:
        use_spmd_for_xla_partitioning = False
    use_spmd_for_xla_partitioning = _execute.make_bool(use_spmd_for_xla_partitioning, 'use_spmd_for_xla_partitioning')
    if tpu_compile_options_proto is None:
        tpu_compile_options_proto = ''
    tpu_compile_options_proto = _execute.make_str(tpu_compile_options_proto, 'tpu_compile_options_proto')
    _inputs_flat = []
    _attrs = ('num_replicas', num_replicas, 'num_cores_per_replica', num_cores_per_replica, 'topology', topology, 'use_tpu', use_tpu, 'device_assignment', device_assignment, 'computation_shape', computation_shape, 'host_compute_core', host_compute_core, 'padding_map', padding_map, 'step_marker_location', step_marker_location, 'allow_soft_placement', allow_soft_placement, 'use_spmd_for_xla_partitioning', use_spmd_for_xla_partitioning, 'tpu_compile_options_proto', tpu_compile_options_proto)
    _result = _execute.execute(b'TPUReplicateMetadata', 0, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    _result = None
    return _result