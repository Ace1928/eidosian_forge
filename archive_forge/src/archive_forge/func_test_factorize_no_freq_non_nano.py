import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('sort', [True, False])
def test_factorize_no_freq_non_nano(self, tz_naive_fixture, sort):
    tz = tz_naive_fixture
    idx = date_range('2016-11-06', freq='h', periods=5, tz=tz)[[0, 4, 1, 3, 2]]
    exp_codes, exp_uniques = idx.factorize(sort=sort)
    res_codes, res_uniques = idx.as_unit('s').factorize(sort=sort)
    tm.assert_numpy_array_equal(res_codes, exp_codes)
    tm.assert_index_equal(res_uniques, exp_uniques.as_unit('s'))
    res_codes, res_uniques = idx.as_unit('s').to_series().factorize(sort=sort)
    tm.assert_numpy_array_equal(res_codes, exp_codes)
    tm.assert_index_equal(res_uniques, exp_uniques.as_unit('s'))