import pytest
import pyarrow as pa
import pyarrow.compute as pc
from pyarrow.compute import field
@pytest.fixture
def table_source():
    table = pa.table({'a': [1, 2, 3], 'b': [4, 5, 6]})
    table_opts = TableSourceNodeOptions(table)
    table_source = Declaration('table_source', options=table_opts)
    return table_source