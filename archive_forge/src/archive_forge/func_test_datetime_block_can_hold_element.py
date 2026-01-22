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
def test_datetime_block_can_hold_element(self):
    block = create_block('datetime', [0])
    assert block._can_hold_element([])
    arr = pd.array(block.values.ravel())
    assert block._can_hold_element(None)
    arr[0] = None
    assert arr[0] is pd.NaT
    vals = [np.datetime64('2010-10-10'), datetime(2010, 10, 10)]
    for val in vals:
        assert block._can_hold_element(val)
        arr[0] = val
    val = date(2010, 10, 10)
    assert not block._can_hold_element(val)
    msg = "value should be a 'Timestamp', 'NaT', or array of those. Got 'date' instead."
    with pytest.raises(TypeError, match=msg):
        arr[0] = val