import pytest
import pyarrow as pa
import pyarrow.compute as pc
from pyarrow.compute import field
def test_declaration_repr(table_source):
    assert 'TableSourceNode' in str(table_source)
    assert 'TableSourceNode' in repr(table_source)