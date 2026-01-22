import re
import numpy as np
import pytest
from pandas._libs.tslibs import Timestamp
import pandas as pd
from pandas import (
from pandas.tests.io.pytables.common import (
from pandas.util import _test_decorators as td
def test_api_default_format(tmp_path, setup_path):
    with ensure_clean_store(setup_path) as store:
        df = DataFrame(1.1 * np.arange(120).reshape((30, 4)), columns=Index(list('ABCD'), dtype=object), index=Index([f'i-{i}' for i in range(30)], dtype=object))
        with pd.option_context('io.hdf.default_format', 'fixed'):
            _maybe_remove(store, 'df')
            store.put('df', df)
            assert not store.get_storer('df').is_table
            msg = 'Can only append to Tables'
            with pytest.raises(ValueError, match=msg):
                store.append('df2', df)
        with pd.option_context('io.hdf.default_format', 'table'):
            _maybe_remove(store, 'df')
            store.put('df', df)
            assert store.get_storer('df').is_table
            _maybe_remove(store, 'df2')
            store.append('df2', df)
            assert store.get_storer('df').is_table
    path = tmp_path / setup_path
    df = DataFrame(1.1 * np.arange(120).reshape((30, 4)), columns=Index(list('ABCD'), dtype=object), index=Index([f'i-{i}' for i in range(30)], dtype=object))
    with pd.option_context('io.hdf.default_format', 'fixed'):
        df.to_hdf(path, key='df')
        with HDFStore(path) as store:
            assert not store.get_storer('df').is_table
        with pytest.raises(ValueError, match=msg):
            df.to_hdf(path, key='df2', append=True)
    with pd.option_context('io.hdf.default_format', 'table'):
        df.to_hdf(path, key='df3')
        with HDFStore(path) as store:
            assert store.get_storer('df3').is_table
        df.to_hdf(path, key='df4', append=True)
        with HDFStore(path) as store:
            assert store.get_storer('df4').is_table