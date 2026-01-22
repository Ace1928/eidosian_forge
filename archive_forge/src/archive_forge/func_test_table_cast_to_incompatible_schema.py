from collections import OrderedDict
from collections.abc import Iterable
import sys
import weakref
import numpy as np
import pytest
import pyarrow as pa
import pyarrow.compute as pc
def test_table_cast_to_incompatible_schema():
    data = [pa.array(range(5)), pa.array([-10, -5, 0, 5, 10])]
    table = pa.Table.from_arrays(data, names=tuple('ab'))
    target_schema1 = pa.schema([pa.field('A', pa.int32()), pa.field('b', pa.int16())])
    target_schema2 = pa.schema([pa.field('a', pa.int32())])
    message = "Target schema's field names are not matching the table's field names:.*"
    with pytest.raises(ValueError, match=message):
        table.cast(target_schema1)
    with pytest.raises(ValueError, match=message):
        table.cast(target_schema2)