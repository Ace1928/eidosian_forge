from itertools import chain
import re
import numpy as np
import pytest
from pandas.errors import SpecificationError
from pandas import (
import pandas._testing as tm
@pytest.mark.filterwarnings('ignore::FutureWarning')
@pytest.mark.parametrize('func, msg', [(['sqrt', 'max'], 'cannot combine transform and aggregation'), ({'foo': np.sqrt, 'bar': 'sum'}, 'cannot perform both aggregation and transformation')])
def test_transform_and_agg_err_series(string_series, func, msg):
    with pytest.raises(ValueError, match=msg):
        with np.errstate(all='ignore'):
            string_series.agg(func)