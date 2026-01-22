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
def test_iset_split_block(self):
    bm = create_mgr('a,b,c: i8; d: f8')
    bm._iset_split_block(0, np.array([0]))
    tm.assert_numpy_array_equal(bm.blklocs, np.array([0, 0, 1, 0], dtype='int64' if IS64 else 'int32'))
    tm.assert_numpy_array_equal(bm.blknos, np.array([0, 0, 0, 1], dtype='int64' if IS64 else 'int32'))
    assert len(bm.blocks) == 2