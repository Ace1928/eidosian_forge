import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.util.hashing import hash_tuples
from pandas.util import (
@pytest.mark.parametrize('val', [5, 'foo', pd.Timestamp('20130101')])
def test_hash_array_errors(val):
    msg = 'must pass a ndarray-like'
    with pytest.raises(TypeError, match=msg):
        hash_array(val)