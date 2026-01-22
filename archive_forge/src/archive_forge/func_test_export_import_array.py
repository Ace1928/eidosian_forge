import contextlib
import ctypes
import gc
import pyarrow as pa
import pytest
@needs_cffi
def test_export_import_array():
    c_schema = ffi.new('struct ArrowSchema*')
    ptr_schema = int(ffi.cast('uintptr_t', c_schema))
    c_array = ffi.new('struct ArrowArray*')
    ptr_array = int(ffi.cast('uintptr_t', c_array))
    gc.collect()
    old_allocated = pa.total_allocated_bytes()
    typ = pa.list_(pa.int32())
    arr = pa.array([[1], [2, 42]], type=typ)
    py_value = arr.to_pylist()
    arr._export_to_c(ptr_array)
    assert pa.total_allocated_bytes() > old_allocated
    del arr
    arr_new = pa.Array._import_from_c(ptr_array, typ)
    assert arr_new.to_pylist() == py_value
    assert arr_new.type == pa.list_(pa.int32())
    assert pa.total_allocated_bytes() > old_allocated
    del arr_new, typ
    assert pa.total_allocated_bytes() == old_allocated
    with assert_array_released:
        pa.Array._import_from_c(ptr_array, pa.list_(pa.int32()))
    arr = pa.array([[1], [2, 42]], type=pa.list_(pa.int32()))
    py_value = arr.to_pylist()
    arr._export_to_c(ptr_array, ptr_schema)
    del arr
    arr_new = pa.Array._import_from_c(ptr_array, ptr_schema)
    assert arr_new.to_pylist() == py_value
    assert arr_new.type == pa.list_(pa.int32())
    assert pa.total_allocated_bytes() > old_allocated
    del arr_new
    assert pa.total_allocated_bytes() == old_allocated
    with assert_schema_released:
        pa.Array._import_from_c(ptr_array, ptr_schema)