from datetime import (
import itertools
import re
import numpy as np
import pytest
from pandas._libs.internals import BlockPlacement
from pandas.compat import IS64
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import is_scalar
import pandas as pd
from pandas import (
import pandas._testing as tm
import pandas.core.algorithms as algos
from pandas.core.arrays import (
from pandas.core.internals import (
from pandas.core.internals.blocks import (
@pytest.mark.parametrize('dtype', [np.int64, np.uint64, np.float64])
def test_interval_can_hold_element_emptylist(self, dtype, element):
    arr = np.array([1, 3, 4], dtype=dtype)
    ii = IntervalIndex.from_breaks(arr)
    blk = new_block(ii._data, BlockPlacement([1]), ndim=2)
    assert blk._can_hold_element([])