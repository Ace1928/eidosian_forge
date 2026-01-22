from collections import OrderedDict
from collections.abc import Iterable
import sys
import weakref
import numpy as np
import pytest
import pyarrow as pa
import pyarrow.compute as pc
def test_table_column_with_duplicates():
    table = pa.table([pa.array([1, 2, 3]), pa.array([4, 5, 6]), pa.array([7, 8, 9])], names=['a', 'b', 'a'])
    with pytest.raises(KeyError, match='Field "a" exists 2 times in schema'):
        table.column('a')