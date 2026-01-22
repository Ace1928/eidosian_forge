import string
import numpy as np
import pytest
import pandas as pd
from pandas import SparseDtype
import pandas._testing as tm
from pandas.core.arrays.sparse import SparseArray
@pytest.mark.parametrize('fill_value', [1, np.nan])
def test_to_coo_nonzero_fill_val_raises(self, fill_value):
    pytest.importorskip('scipy')
    df = pd.DataFrame({'A': SparseArray([fill_value, fill_value, fill_value, 2], fill_value=fill_value), 'B': SparseArray([fill_value, 2, fill_value, fill_value], fill_value=fill_value)})
    with pytest.raises(ValueError, match='fill value must be 0'):
        df.sparse.to_coo()