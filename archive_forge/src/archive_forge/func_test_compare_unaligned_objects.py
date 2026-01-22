import numpy as np
import pytest
from pandas.compat.numpy import np_version_gte1p25
import pandas as pd
import pandas._testing as tm
def test_compare_unaligned_objects():
    msg = 'Can only compare identically-labeled \\(both index and columns\\) DataFrame objects'
    with pytest.raises(ValueError, match=msg):
        df1 = pd.DataFrame([1, 2, 3], index=['a', 'b', 'c'])
        df2 = pd.DataFrame([1, 2, 3], index=['a', 'b', 'd'])
        df1.compare(df2)
    msg = 'Can only compare identically-labeled \\(both index and columns\\) DataFrame objects'
    with pytest.raises(ValueError, match=msg):
        df1 = pd.DataFrame(np.ones((3, 3)))
        df2 = pd.DataFrame(np.zeros((2, 1)))
        df1.compare(df2)