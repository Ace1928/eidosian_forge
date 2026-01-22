import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('cls', [Categorical, CategoricalIndex])
@pytest.mark.parametrize('values', [[1, np.nan], [Timestamp('2000'), NaT]])
def test_astype_nan_to_int(self, cls, values):
    obj = cls(values)
    msg = 'Cannot (cast|convert)'
    with pytest.raises((ValueError, TypeError), match=msg):
        obj.astype(int)