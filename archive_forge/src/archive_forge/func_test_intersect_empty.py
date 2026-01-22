import operator
import numpy as np
import pytest
import pandas._libs.sparse as splib
import pandas.util._test_decorators as td
from pandas import Series
import pandas._testing as tm
from pandas.core.arrays.sparse import (
def test_intersect_empty(self):
    xindex = IntIndex(4, np.array([], dtype=np.int32))
    yindex = IntIndex(4, np.array([2, 3], dtype=np.int32))
    assert xindex.intersect(yindex).equals(xindex)
    assert yindex.intersect(xindex).equals(xindex)
    xindex = xindex.to_block_index()
    yindex = yindex.to_block_index()
    assert xindex.intersect(yindex).equals(xindex)
    assert yindex.intersect(xindex).equals(xindex)