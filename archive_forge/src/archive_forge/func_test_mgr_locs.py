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
def test_mgr_locs(self, fblock):
    assert isinstance(fblock.mgr_locs, BlockPlacement)
    tm.assert_numpy_array_equal(fblock.mgr_locs.as_array, np.array([0, 2, 4], dtype=np.intp))