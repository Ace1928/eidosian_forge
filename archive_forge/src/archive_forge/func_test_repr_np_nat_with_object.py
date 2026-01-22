from datetime import (
from io import StringIO
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
import pandas.io.formats.format as fmt
@pytest.mark.parametrize('arg', [np.datetime64, np.timedelta64])
@pytest.mark.parametrize('box, expected', [[Series, '0    NaT\ndtype: object'], [DataFrame, '     0\n0  NaT']])
def test_repr_np_nat_with_object(self, arg, box, expected):
    result = repr(box([arg('NaT')], dtype=object))
    assert result == expected