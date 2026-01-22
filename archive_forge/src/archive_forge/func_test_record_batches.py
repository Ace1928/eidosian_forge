import hypothesis as h
import pyarrow as pa
import pyarrow.tests.strategies as past
@h.given(past.all_record_batches)
def test_record_batches(record_bath):
    assert isinstance(record_bath, pa.lib.RecordBatch)