import contextlib
import json
from pathlib import Path
import numpy as np
import pandas
import pytest
from pandas._testing import ensure_clean
import modin.experimental.pandas as pd
from modin.config import AsyncReadMode, Engine
from modin.tests.pandas.utils import (
from modin.tests.test_utils import warns_that_defaulting_to_pandas
from modin.utils import try_cast_to_pandas
@pytest.mark.skipif(Engine.get() not in ('Ray', 'Unidist', 'Dask'), reason=f'{Engine.get()} does not have experimental API')
@pytest.mark.parametrize('pathlike', [False, True])
@pytest.mark.parametrize('compression', [None, 'gzip'])
@pytest.mark.parametrize('filename', ['test_default_to_pickle.pkl', 'test_to_pickle*.pkl'])
@pytest.mark.parametrize('read_func', ['read_pickle_glob', 'read_pickle_distributed'])
@pytest.mark.parametrize('to_func', ['to_pickle_glob', 'to_pickle_distributed'])
def test_distributed_pickling(tmp_path, filename, compression, pathlike, read_func, to_func):
    data = test_data['int_data']
    df = pd.DataFrame(data)
    filename_param = filename
    if compression:
        filename = f'{filename}.gz'
    filename = Path(filename) if pathlike else filename
    with warns_that_defaulting_to_pandas() if filename_param == 'test_default_to_pickle.pkl' else contextlib.nullcontext():
        getattr(df.modin, to_func)(str(tmp_path / filename), compression=compression)
        pickled_df = getattr(pd, read_func)(str(tmp_path / filename), compression=compression)
    df_equals(pickled_df, df)