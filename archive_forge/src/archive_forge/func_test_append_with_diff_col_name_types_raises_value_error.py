import datetime
from io import BytesIO
import re
import numpy as np
import pytest
from pandas import (
from pandas.tests.io.pytables.common import ensure_clean_store
from pandas.io.pytables import (
def test_append_with_diff_col_name_types_raises_value_error(setup_path):
    df = DataFrame(np.random.default_rng(2).standard_normal((10, 1)))
    df2 = DataFrame({'a': np.random.default_rng(2).standard_normal(10)})
    df3 = DataFrame({(1, 2): np.random.default_rng(2).standard_normal(10)})
    df4 = DataFrame({('1', 2): np.random.default_rng(2).standard_normal(10)})
    df5 = DataFrame({('1', 2, object): np.random.default_rng(2).standard_normal(10)})
    with ensure_clean_store(setup_path) as store:
        name = 'df_diff_valerror'
        store.append(name, df)
        for d in (df2, df3, df4, df5):
            msg = re.escape('cannot match existing table structure for [0] on appending data')
            with pytest.raises(ValueError, match=msg):
                store.append(name, d)