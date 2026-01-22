import contextlib
import ctypes
import gc
import pyarrow as pa
import pytest
def check_export_import_batch(batch_factory):
    c_schema = ffi.new('struct ArrowSchema*')
    ptr_schema = int(ffi.cast('uintptr_t', c_schema))
    c_array = ffi.new('struct ArrowArray*')
    ptr_array = int(ffi.cast('uintptr_t', c_array))
    gc.collect()
    old_allocated = pa.total_allocated_bytes()
    batch = batch_factory()
    schema = batch.schema
    py_value = batch.to_pydict()
    batch._export_to_c(ptr_array)
    assert pa.total_allocated_bytes() > old_allocated
    del batch
    batch_new = pa.RecordBatch._import_from_c(ptr_array, schema)
    assert batch_new.to_pydict() == py_value
    assert batch_new.schema == schema
    assert pa.total_allocated_bytes() > old_allocated
    del batch_new, schema
    assert pa.total_allocated_bytes() == old_allocated
    with assert_array_released:
        pa.RecordBatch._import_from_c(ptr_array, make_schema())
    batch = batch_factory()
    py_value = batch.to_pydict()
    batch._export_to_c(ptr_array, ptr_schema)
    del batch
    batch_new = pa.RecordBatch._import_from_c(ptr_array, ptr_schema)
    assert batch_new.to_pydict() == py_value
    assert batch_new.schema == batch_factory().schema
    assert pa.total_allocated_bytes() > old_allocated
    del batch_new
    assert pa.total_allocated_bytes() == old_allocated
    with assert_schema_released:
        pa.RecordBatch._import_from_c(ptr_array, ptr_schema)
    pa.int32()._export_to_c(ptr_schema)
    batch_factory()._export_to_c(ptr_array)
    with pytest.raises(ValueError, match='ArrowSchema describes non-struct type'):
        pa.RecordBatch._import_from_c(ptr_array, ptr_schema)
    with assert_schema_released:
        pa.RecordBatch._import_from_c(ptr_array, ptr_schema)