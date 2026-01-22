from copy import deepcopy
from operator import methodcaller
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_unexpected_keyword(self):
    df = DataFrame(np.random.default_rng(2).standard_normal((5, 2)), columns=['jim', 'joe'])
    ca = pd.Categorical([0, 0, 2, 2, 3, np.nan])
    ts = df['joe'].copy()
    ts[2] = np.nan
    msg = 'unexpected keyword'
    with pytest.raises(TypeError, match=msg):
        df.drop('joe', axis=1, in_place=True)
    with pytest.raises(TypeError, match=msg):
        df.reindex([1, 0], inplace=True)
    with pytest.raises(TypeError, match=msg):
        ca.fillna(0, inplace=True)
    with pytest.raises(TypeError, match=msg):
        ts.fillna(0, in_place=True)