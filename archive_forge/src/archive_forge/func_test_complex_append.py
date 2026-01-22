import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.io.pytables.common import ensure_clean_store
from pandas.io.pytables import read_hdf
def test_complex_append(setup_path):
    df = DataFrame({'a': np.random.default_rng(2).standard_normal(100).astype(np.complex128), 'b': np.random.default_rng(2).standard_normal(100)})
    with ensure_clean_store(setup_path) as store:
        store.append('df', df, data_columns=['b'])
        store.append('df', df)
        result = store.select('df')
        tm.assert_frame_equal(pd.concat([df, df], axis=0), result)