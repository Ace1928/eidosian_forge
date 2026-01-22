import operator
import numpy as np
import pytest
import pandas._libs.sparse as splib
import pandas.util._test_decorators as td
from pandas import Series
import pandas._testing as tm
from pandas.core.arrays.sparse import (
def test_to_block_index(self, cases, test_length):
    xloc, xlen, yloc, ylen, _, _ = cases
    xindex = BlockIndex(test_length, xloc, xlen)
    yindex = BlockIndex(test_length, yloc, ylen)
    xbindex = xindex.to_int_index().to_block_index()
    ybindex = yindex.to_int_index().to_block_index()
    assert isinstance(xbindex, BlockIndex)
    assert xbindex.equals(xindex)
    assert ybindex.equals(yindex)