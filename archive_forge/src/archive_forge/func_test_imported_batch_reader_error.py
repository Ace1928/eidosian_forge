import contextlib
import ctypes
import gc
import pyarrow as pa
import pytest
@needs_cffi
def test_imported_batch_reader_error():
    c_stream = ffi.new('struct ArrowArrayStream*')
    ptr_stream = int(ffi.cast('uintptr_t', c_stream))
    schema = pa.schema([('foo', pa.int32())])
    batches = [pa.record_batch([[1, 2, 3]], schema=schema), pa.record_batch([[4, 5, 6]], schema=schema)]
    buf = make_serialized(schema, batches)
    reader = pa.ipc.open_stream(buf[:-16])
    reader._export_to_c(ptr_stream)
    del reader
    reader_new = pa.RecordBatchReader._import_from_c(ptr_stream)
    batch = reader_new.read_next_batch()
    assert batch == batches[0]
    with pytest.raises(OSError, match='Expected to be able to read 16 bytes for message body, got 8'):
        reader_new.read_next_batch()
    reader = pa.ipc.open_stream(buf[:-16])
    reader._export_to_c(ptr_stream)
    del reader
    reader_new = pa.RecordBatchReader._import_from_c(ptr_stream)
    with pytest.raises(OSError, match='Expected to be able to read 16 bytes for message body, got 8'):
        reader_new.read_all()