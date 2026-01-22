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
@pytest.mark.skipif(Engine.get() not in ('Ray', 'Unidist', 'Dask'), reason=f'{Engine.get()} does not have experimental glob API')
@pytest.mark.parametrize('path', ['s3://modin-test/modin-bugs/multiple_csv/test_data*.csv'])
def test_read_multiple_csv_cloud_store(path, s3_resource, s3_storage_options):

    def _pandas_read_csv_glob(path, storage_options):
        pandas_dfs = [pandas.read_csv(f'{path.lower().split('*')[0]}{i}.csv', storage_options=storage_options) for i in range(2)]
        return pandas.concat(pandas_dfs).reset_index(drop=True)
    eval_general(pd, pandas, lambda module, **kwargs: pd.read_csv_glob(path, **kwargs).reset_index(drop=True) if hasattr(module, 'read_csv_glob') else _pandas_read_csv_glob(path, **kwargs), storage_options=s3_storage_options)