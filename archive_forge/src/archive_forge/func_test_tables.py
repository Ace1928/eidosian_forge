import hypothesis as h
import pyarrow as pa
import pyarrow.tests.strategies as past
@h.given(past.all_tables)
def test_tables(table):
    assert isinstance(table, pa.lib.Table)