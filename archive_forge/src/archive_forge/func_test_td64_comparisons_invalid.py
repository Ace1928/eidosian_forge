from datetime import (
import numpy as np
import pytest
from pandas.errors import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import NumpyExtensionArray
from pandas.tests.arithmetic.common import (
@pytest.mark.parametrize('invalid', [345600000000000, 'a', Timestamp('2021-01-01'), Timestamp('2021-01-01').now('UTC'), Timestamp('2021-01-01').now().to_datetime64(), Timestamp('2021-01-01').now().to_pydatetime(), Timestamp('2021-01-01').date(), np.array(4)])
def test_td64_comparisons_invalid(self, box_with_array, invalid):
    box = box_with_array
    rng = timedelta_range('1 days', periods=10)
    obj = tm.box_expected(rng, box)
    assert_invalid_comparison(obj, invalid, box)