import hypothesis as h
import pyarrow as pa
import pyarrow.tests.strategies as past
@h.given(past.all_chunked_arrays)
def test_chunked_arrays(chunked_array):
    assert isinstance(chunked_array, pa.lib.ChunkedArray)