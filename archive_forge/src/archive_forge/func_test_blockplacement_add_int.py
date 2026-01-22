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
@pytest.mark.parametrize('val, inc, expected', [(slice(0, 0), 0, []), (slice(1, 4), 0, [1, 2, 3]), (slice(3, 0, -1), 0, [3, 2, 1]), ([1, 2, 4], 0, [1, 2, 4]), (slice(0, 0), 10, []), (slice(1, 4), 10, [11, 12, 13]), (slice(3, 0, -1), 10, [13, 12, 11]), ([1, 2, 4], 10, [11, 12, 14]), (slice(0, 0), -1, []), (slice(1, 4), -1, [0, 1, 2]), ([1, 2, 4], -1, [0, 1, 3])])
def test_blockplacement_add_int(self, val, inc, expected):
    assert list(BlockPlacement(val).add(inc)) == expected