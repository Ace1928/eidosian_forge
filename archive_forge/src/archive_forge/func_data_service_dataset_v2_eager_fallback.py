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
def data_service_dataset_v2_eager_fallback(dataset_id: _atypes.TensorFuzzingAnnotation[_atypes.Int64], processing_mode: _atypes.TensorFuzzingAnnotation[_atypes.String], address: _atypes.TensorFuzzingAnnotation[_atypes.String], protocol: _atypes.TensorFuzzingAnnotation[_atypes.String], job_name: _atypes.TensorFuzzingAnnotation[_atypes.String], consumer_index: _atypes.TensorFuzzingAnnotation[_atypes.Int64], num_consumers: _atypes.TensorFuzzingAnnotation[_atypes.Int64], max_outstanding_requests: _atypes.TensorFuzzingAnnotation[_atypes.Int64], iteration_counter: _atypes.TensorFuzzingAnnotation[_atypes.Resource], output_types, output_shapes, task_refresh_interval_hint_ms: int, data_transfer_protocol: str, target_workers: str, cross_trainer_cache_options: str, name, ctx) -> _atypes.TensorFuzzingAnnotation[_atypes.Variant]:
    if not isinstance(output_types, (list, tuple)):
        raise TypeError("Expected list for 'output_types' argument to 'data_service_dataset_v2' Op, not %r." % output_types)
    output_types = [_execute.make_type(_t, 'output_types') for _t in output_types]
    if not isinstance(output_shapes, (list, tuple)):
        raise TypeError("Expected list for 'output_shapes' argument to 'data_service_dataset_v2' Op, not %r." % output_shapes)
    output_shapes = [_execute.make_shape(_s, 'output_shapes') for _s in output_shapes]
    if task_refresh_interval_hint_ms is None:
        task_refresh_interval_hint_ms = -1
    task_refresh_interval_hint_ms = _execute.make_int(task_refresh_interval_hint_ms, 'task_refresh_interval_hint_ms')
    if data_transfer_protocol is None:
        data_transfer_protocol = ''
    data_transfer_protocol = _execute.make_str(data_transfer_protocol, 'data_transfer_protocol')
    if target_workers is None:
        target_workers = 'AUTO'
    target_workers = _execute.make_str(target_workers, 'target_workers')
    if cross_trainer_cache_options is None:
        cross_trainer_cache_options = ''
    cross_trainer_cache_options = _execute.make_str(cross_trainer_cache_options, 'cross_trainer_cache_options')
    dataset_id = _ops.convert_to_tensor(dataset_id, _dtypes.int64)
    processing_mode = _ops.convert_to_tensor(processing_mode, _dtypes.string)
    address = _ops.convert_to_tensor(address, _dtypes.string)
    protocol = _ops.convert_to_tensor(protocol, _dtypes.string)
    job_name = _ops.convert_to_tensor(job_name, _dtypes.string)
    consumer_index = _ops.convert_to_tensor(consumer_index, _dtypes.int64)
    num_consumers = _ops.convert_to_tensor(num_consumers, _dtypes.int64)
    max_outstanding_requests = _ops.convert_to_tensor(max_outstanding_requests, _dtypes.int64)
    iteration_counter = _ops.convert_to_tensor(iteration_counter, _dtypes.resource)
    _inputs_flat = [dataset_id, processing_mode, address, protocol, job_name, consumer_index, num_consumers, max_outstanding_requests, iteration_counter]
    _attrs = ('task_refresh_interval_hint_ms', task_refresh_interval_hint_ms, 'output_types', output_types, 'output_shapes', output_shapes, 'data_transfer_protocol', data_transfer_protocol, 'target_workers', target_workers, 'cross_trainer_cache_options', cross_trainer_cache_options)
    _result = _execute.execute(b'DataServiceDatasetV2', 1, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('DataServiceDatasetV2', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result