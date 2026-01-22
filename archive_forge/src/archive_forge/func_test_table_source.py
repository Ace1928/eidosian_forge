import pytest
import pyarrow as pa
import pyarrow.compute as pc
from pyarrow.compute import field
def test_table_source():
    with pytest.raises(TypeError):
        TableSourceNodeOptions(pa.record_batch([pa.array([1, 2, 3])], ['a']))
    table_source = TableSourceNodeOptions(None)
    decl = Declaration('table_source', table_source)
    with pytest.raises(ValueError, match='TableSourceNode requires table which is not null'):
        _ = decl.to_table()