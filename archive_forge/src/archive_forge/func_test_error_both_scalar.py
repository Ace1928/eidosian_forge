import operator
import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
from pandas.arrays import BooleanArray
from pandas.core.ops.mask_ops import (
from pandas.tests.extension.base import BaseOpsUtil
@pytest.mark.parametrize('operation', [kleene_or, kleene_xor, kleene_and])
def test_error_both_scalar(operation):
    msg = 'Either `left` or `right` need to be a np\\.ndarray.'
    with pytest.raises(TypeError, match=msg):
        operation(True, True, np.zeros(1), np.zeros(1))