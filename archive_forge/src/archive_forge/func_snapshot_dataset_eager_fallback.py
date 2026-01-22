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
def snapshot_dataset_eager_fallback(input_dataset: _atypes.TensorFuzzingAnnotation[_atypes.Variant], path: _atypes.TensorFuzzingAnnotation[_atypes.String], output_types, output_shapes, compression: str, reader_path_prefix: str, writer_path_prefix: str, shard_size_bytes: int, pending_snapshot_expiry_seconds: int, num_reader_threads: int, reader_buffer_size: int, num_writer_threads: int, writer_buffer_size: int, shuffle_on_read: bool, seed: int, seed2: int, mode: str, snapshot_name: str, name, ctx) -> _atypes.TensorFuzzingAnnotation[_atypes.Variant]:
    if not isinstance(output_types, (list, tuple)):
        raise TypeError("Expected list for 'output_types' argument to 'snapshot_dataset' Op, not %r." % output_types)
    output_types = [_execute.make_type(_t, 'output_types') for _t in output_types]
    if not isinstance(output_shapes, (list, tuple)):
        raise TypeError("Expected list for 'output_shapes' argument to 'snapshot_dataset' Op, not %r." % output_shapes)
    output_shapes = [_execute.make_shape(_s, 'output_shapes') for _s in output_shapes]
    if compression is None:
        compression = ''
    compression = _execute.make_str(compression, 'compression')
    if reader_path_prefix is None:
        reader_path_prefix = ''
    reader_path_prefix = _execute.make_str(reader_path_prefix, 'reader_path_prefix')
    if writer_path_prefix is None:
        writer_path_prefix = ''
    writer_path_prefix = _execute.make_str(writer_path_prefix, 'writer_path_prefix')
    if shard_size_bytes is None:
        shard_size_bytes = 10737418240
    shard_size_bytes = _execute.make_int(shard_size_bytes, 'shard_size_bytes')
    if pending_snapshot_expiry_seconds is None:
        pending_snapshot_expiry_seconds = 86400
    pending_snapshot_expiry_seconds = _execute.make_int(pending_snapshot_expiry_seconds, 'pending_snapshot_expiry_seconds')
    if num_reader_threads is None:
        num_reader_threads = 1
    num_reader_threads = _execute.make_int(num_reader_threads, 'num_reader_threads')
    if reader_buffer_size is None:
        reader_buffer_size = 1
    reader_buffer_size = _execute.make_int(reader_buffer_size, 'reader_buffer_size')
    if num_writer_threads is None:
        num_writer_threads = 1
    num_writer_threads = _execute.make_int(num_writer_threads, 'num_writer_threads')
    if writer_buffer_size is None:
        writer_buffer_size = 1
    writer_buffer_size = _execute.make_int(writer_buffer_size, 'writer_buffer_size')
    if shuffle_on_read is None:
        shuffle_on_read = False
    shuffle_on_read = _execute.make_bool(shuffle_on_read, 'shuffle_on_read')
    if seed is None:
        seed = 0
    seed = _execute.make_int(seed, 'seed')
    if seed2 is None:
        seed2 = 0
    seed2 = _execute.make_int(seed2, 'seed2')
    if mode is None:
        mode = 'auto'
    mode = _execute.make_str(mode, 'mode')
    if snapshot_name is None:
        snapshot_name = ''
    snapshot_name = _execute.make_str(snapshot_name, 'snapshot_name')
    input_dataset = _ops.convert_to_tensor(input_dataset, _dtypes.variant)
    path = _ops.convert_to_tensor(path, _dtypes.string)
    _inputs_flat = [input_dataset, path]
    _attrs = ('output_types', output_types, 'output_shapes', output_shapes, 'compression', compression, 'reader_path_prefix', reader_path_prefix, 'writer_path_prefix', writer_path_prefix, 'shard_size_bytes', shard_size_bytes, 'pending_snapshot_expiry_seconds', pending_snapshot_expiry_seconds, 'num_reader_threads', num_reader_threads, 'reader_buffer_size', reader_buffer_size, 'num_writer_threads', num_writer_threads, 'writer_buffer_size', writer_buffer_size, 'shuffle_on_read', shuffle_on_read, 'seed', seed, 'seed2', seed2, 'mode', mode, 'snapshot_name', snapshot_name)
    _result = _execute.execute(b'SnapshotDataset', 1, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('SnapshotDataset', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result