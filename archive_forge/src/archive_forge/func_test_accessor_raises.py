import string
import numpy as np
import pytest
import pandas as pd
from pandas import SparseDtype
import pandas._testing as tm
from pandas.core.arrays.sparse import SparseArray
def test_accessor_raises(self):
    df = pd.DataFrame({'A': [0, 1]})
    with pytest.raises(AttributeError, match='sparse'):
        df.sparse