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
@pytest.mark.parametrize('arr', [[], [-1], [-1, -2, -3], [-10], [-1], [-1, 0, 1, 2], [-2, 0, 2, 4], [1, 0, -1], [1, 1, 1]])
def test_not_slice_like_arrays(self, arr):
    assert not BlockPlacement(arr).is_slice_like