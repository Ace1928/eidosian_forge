import contextlib
import datetime as dt
import hashlib
import tempfile
import time
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.io.pytables.common import (
from pandas.io.pytables import (
def test_create_table_index(setup_path):
    with ensure_clean_store(setup_path) as store:

        def col(t, column):
            return getattr(store.get_storer(t).table.cols, column)
        df = DataFrame(np.random.default_rng(2).standard_normal((10, 4)), columns=Index(list('ABCD'), dtype=object), index=date_range('2000-01-01', periods=10, freq='B'))
        df['string'] = 'foo'
        df['string2'] = 'bar'
        store.append('f', df, data_columns=['string', 'string2'])
        assert col('f', 'index').is_indexed is True
        assert col('f', 'string').is_indexed is True
        assert col('f', 'string2').is_indexed is True
        store.append('f2', df, index=['string'], data_columns=['string', 'string2'])
        assert col('f2', 'index').is_indexed is False
        assert col('f2', 'string').is_indexed is True
        assert col('f2', 'string2').is_indexed is False
        _maybe_remove(store, 'f2')
        store.put('f2', df)
        msg = 'cannot create table index on a Fixed format store'
        with pytest.raises(TypeError, match=msg):
            store.create_table_index('f2')