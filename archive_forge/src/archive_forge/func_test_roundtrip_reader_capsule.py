import contextlib
import ctypes
import gc
import pyarrow as pa
import pytest
@pytest.mark.parametrize('constructor', [pa.RecordBatchReader.from_batches, lambda schema, batches: pa.Table.from_batches(batches, schema)], ids=['recordbatchreader', 'table'])
def test_roundtrip_reader_capsule(constructor):
    batches = make_batches()
    schema = batches[0].schema
    gc.collect()
    old_allocated = pa.total_allocated_bytes()
    obj = constructor(schema, batches)
    capsule = obj.__arrow_c_stream__()
    assert PyCapsule_IsValid(capsule, b'arrow_array_stream') == 1
    imported_reader = pa.RecordBatchReader._import_from_c_capsule(capsule)
    assert imported_reader.schema == schema
    imported_batches = list(imported_reader)
    assert len(imported_batches) == len(batches)
    for batch, expected in zip(imported_batches, batches):
        assert batch.equals(expected)
    del obj, imported_reader, batch, expected, imported_batches
    assert pa.total_allocated_bytes() == old_allocated
    obj = constructor(schema, batches)
    bad_schema = pa.schema({'ints': pa.int32()})
    with pytest.raises(NotImplementedError):
        obj.__arrow_c_stream__(bad_schema.__arrow_c_schema__())
    matching_schema = pa.schema({'ints': pa.list_(pa.int32())})
    capsule = obj.__arrow_c_stream__(matching_schema.__arrow_c_schema__())
    imported_reader = pa.RecordBatchReader._import_from_c_capsule(capsule)
    assert imported_reader.schema == matching_schema
    for batch, expected in zip(imported_reader, batches):
        assert batch.equals(expected)