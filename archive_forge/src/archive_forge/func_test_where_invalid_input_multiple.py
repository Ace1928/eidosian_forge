from datetime import datetime
from hypothesis import given
import numpy as np
import pytest
from pandas.core.dtypes.common import is_scalar
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas._testing._hypothesis import OPTIONAL_ONE_OF_ALL
@pytest.mark.parametrize('cond', [[[0, 1], [1, 0], [1, 1]], Series([[0, 2], [5, 0], [4, 7]]), [['False', 'True'], ['True', 'False'], ['True', 'True']], DataFrame({'a': [2, 5, 7], 'b': [4, 8, 9]}), [[pd.NaT, Timestamp('2017-01-01')], [Timestamp('2017-01-02'), pd.NaT], [Timestamp('2017-01-03'), Timestamp('2017-01-03')]]])
def test_where_invalid_input_multiple(self, cond):
    df = DataFrame({'a': [1, 2, 3], 'b': [2, 2, 2]})
    msg = 'Boolean array expected for the condition'
    with pytest.raises(ValueError, match=msg):
        df.where(cond)