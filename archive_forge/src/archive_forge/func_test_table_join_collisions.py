from collections import OrderedDict
from collections.abc import Iterable
import sys
import weakref
import numpy as np
import pytest
import pyarrow as pa
import pyarrow.compute as pc
@pytest.mark.acero
def test_table_join_collisions():
    t1 = pa.table({'colA': [1, 2, 6], 'colB': [10, 20, 60], 'colVals': ['a', 'b', 'f']})
    t2 = pa.table({'colA': [99, 2, 1], 'colB': [99, 20, 10], 'colVals': ['Z', 'B', 'A']})
    result = t1.join(t2, 'colA', join_type='full outer')
    assert result.combine_chunks().sort_by('colA') == pa.table([[1, 2, 6, 99], [10, 20, 60, None], ['a', 'b', 'f', None], [10, 20, None, 99], ['A', 'B', None, 'Z']], names=['colA', 'colB', 'colVals', 'colB', 'colVals'])