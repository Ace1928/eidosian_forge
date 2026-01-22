import operator
import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
from pandas.arrays import BooleanArray
from pandas.core.ops.mask_ops import (
from pandas.tests.extension.base import BaseOpsUtil
def test_logical_length_mismatch_raises(self, all_logical_operators):
    op_name = all_logical_operators
    a = pd.array([True, False, None], dtype='boolean')
    msg = 'Lengths must match'
    with pytest.raises(ValueError, match=msg):
        getattr(a, op_name)([True, False])
    with pytest.raises(ValueError, match=msg):
        getattr(a, op_name)(np.array([True, False]))
    with pytest.raises(ValueError, match=msg):
        getattr(a, op_name)(pd.array([True, False], dtype='boolean'))