from datetime import datetime
from hypothesis import given
import numpy as np
import pytest
from pandas.core.dtypes.common import is_scalar
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas._testing._hypothesis import OPTIONAL_ONE_OF_ALL
@pytest.mark.parametrize('cond', [[[1], [0], [1]], Series([[2], [5], [7]]), DataFrame({'a': [2, 5, 7]}), [['True'], ['False'], ['True']], [[Timestamp('2017-01-01')], [pd.NaT], [Timestamp('2017-01-02')]]])
def test_where_invalid_input_single(self, cond):
    df = DataFrame({'a': [1, 2, 3]})
    msg = 'Boolean array expected for the condition'
    with pytest.raises(ValueError, match=msg):
        df.where(cond)