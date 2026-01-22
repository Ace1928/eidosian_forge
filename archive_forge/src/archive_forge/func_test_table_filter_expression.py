from collections import OrderedDict
from collections.abc import Iterable
import sys
import weakref
import numpy as np
import pytest
import pyarrow as pa
import pyarrow.compute as pc
@pytest.mark.acero
def test_table_filter_expression():
    t1 = pa.table({'colA': [1, 2, 6], 'colB': [10, 20, 60], 'colVals': ['a', 'b', 'f']})
    t2 = pa.table({'colA': [99, 2, 1], 'colB': [99, 20, 10], 'colVals': ['Z', 'B', 'A']})
    t3 = pa.concat_tables([t1, t2])
    result = t3.filter(pc.field('colA') < 10)
    assert result.combine_chunks() == pa.table({'colA': [1, 2, 6, 2, 1], 'colB': [10, 20, 60, 20, 10], 'colVals': ['a', 'b', 'f', 'B', 'A']})