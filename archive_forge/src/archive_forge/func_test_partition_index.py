from datetime import datetime
import re
import numpy as np
import pytest
import pandas as pd
from pandas import (
from pandas.tests.strings import (
@pytest.mark.parametrize('method, expand, exp, exp_levels', [['partition', False, np.array([('a', '_', 'b_c'), ('c', '_', 'd_e'), ('f', '_', 'g_h'), np.nan, None], dtype=object), 1], ['rpartition', False, np.array([('a_b', '_', 'c'), ('c_d', '_', 'e'), ('f_g', '_', 'h'), np.nan, None], dtype=object), 1]])
def test_partition_index(method, expand, exp, exp_levels):
    values = Index(['a_b_c', 'c_d_e', 'f_g_h', np.nan, None])
    result = getattr(values.str, method)('_', expand=expand)
    exp = Index(exp)
    tm.assert_index_equal(result, exp)
    assert result.nlevels == exp_levels