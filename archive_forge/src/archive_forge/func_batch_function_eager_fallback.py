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
def batch_function_eager_fallback(in_tensors, captured_tensors, f, num_batch_threads: int, max_batch_size: int, batch_timeout_micros: int, Tout, max_enqueued_batches: int, allowed_batch_sizes, container: str, shared_name: str, batching_queue: str, low_priority_max_batch_size: int, low_priority_batch_timeout_micros: int, low_priority_allowed_batch_sizes, low_priority_max_enqueued_batches: int, enable_large_batch_splitting: bool, name, ctx):
    num_batch_threads = _execute.make_int(num_batch_threads, 'num_batch_threads')
    max_batch_size = _execute.make_int(max_batch_size, 'max_batch_size')
    batch_timeout_micros = _execute.make_int(batch_timeout_micros, 'batch_timeout_micros')
    if not isinstance(Tout, (list, tuple)):
        raise TypeError("Expected list for 'Tout' argument to 'batch_function' Op, not %r." % Tout)
    Tout = [_execute.make_type(_t, 'Tout') for _t in Tout]
    if max_enqueued_batches is None:
        max_enqueued_batches = 10
    max_enqueued_batches = _execute.make_int(max_enqueued_batches, 'max_enqueued_batches')
    if allowed_batch_sizes is None:
        allowed_batch_sizes = []
    if not isinstance(allowed_batch_sizes, (list, tuple)):
        raise TypeError("Expected list for 'allowed_batch_sizes' argument to 'batch_function' Op, not %r." % allowed_batch_sizes)
    allowed_batch_sizes = [_execute.make_int(_i, 'allowed_batch_sizes') for _i in allowed_batch_sizes]
    if container is None:
        container = ''
    container = _execute.make_str(container, 'container')
    if shared_name is None:
        shared_name = ''
    shared_name = _execute.make_str(shared_name, 'shared_name')
    if batching_queue is None:
        batching_queue = ''
    batching_queue = _execute.make_str(batching_queue, 'batching_queue')
    if low_priority_max_batch_size is None:
        low_priority_max_batch_size = 0
    low_priority_max_batch_size = _execute.make_int(low_priority_max_batch_size, 'low_priority_max_batch_size')
    if low_priority_batch_timeout_micros is None:
        low_priority_batch_timeout_micros = 0
    low_priority_batch_timeout_micros = _execute.make_int(low_priority_batch_timeout_micros, 'low_priority_batch_timeout_micros')
    if low_priority_allowed_batch_sizes is None:
        low_priority_allowed_batch_sizes = []
    if not isinstance(low_priority_allowed_batch_sizes, (list, tuple)):
        raise TypeError("Expected list for 'low_priority_allowed_batch_sizes' argument to 'batch_function' Op, not %r." % low_priority_allowed_batch_sizes)
    low_priority_allowed_batch_sizes = [_execute.make_int(_i, 'low_priority_allowed_batch_sizes') for _i in low_priority_allowed_batch_sizes]
    if low_priority_max_enqueued_batches is None:
        low_priority_max_enqueued_batches = 0
    low_priority_max_enqueued_batches = _execute.make_int(low_priority_max_enqueued_batches, 'low_priority_max_enqueued_batches')
    if enable_large_batch_splitting is None:
        enable_large_batch_splitting = False
    enable_large_batch_splitting = _execute.make_bool(enable_large_batch_splitting, 'enable_large_batch_splitting')
    _attr_Tin, in_tensors = _execute.convert_to_mixed_eager_tensors(in_tensors, ctx)
    _attr_Tcaptured, captured_tensors = _execute.convert_to_mixed_eager_tensors(captured_tensors, ctx)
    _inputs_flat = list(in_tensors) + list(captured_tensors)
    _attrs = ('f', f, 'num_batch_threads', num_batch_threads, 'max_batch_size', max_batch_size, 'batch_timeout_micros', batch_timeout_micros, 'max_enqueued_batches', max_enqueued_batches, 'allowed_batch_sizes', allowed_batch_sizes, 'container', container, 'shared_name', shared_name, 'batching_queue', batching_queue, 'low_priority_max_batch_size', low_priority_max_batch_size, 'low_priority_batch_timeout_micros', low_priority_batch_timeout_micros, 'low_priority_allowed_batch_sizes', low_priority_allowed_batch_sizes, 'low_priority_max_enqueued_batches', low_priority_max_enqueued_batches, 'Tin', _attr_Tin, 'Tcaptured', _attr_Tcaptured, 'Tout', Tout, 'enable_large_batch_splitting', enable_large_batch_splitting)
    _result = _execute.execute(b'BatchFunction', len(Tout), inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('BatchFunction', _inputs_flat, _attrs, _result)
    return _result