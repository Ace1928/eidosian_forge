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
@pytest.mark.parametrize('storage_options_extra', [{'anon': False}, {'anon': True}, {'key': '123', 'secret': '123'}])
def test_read_multiple_csv_s3_storage_opts(s3_resource, s3_storage_options, storage_options_extra):
    s3_path = 's3://modin-test/modin-bugs/multiple_csv/'

    def _pandas_read_csv_glob(path, storage_options):
        pandas_df = pandas.concat([pandas.read_csv(f'{s3_path}test_data{i}.csv', storage_options=storage_options) for i in range(2)]).reset_index(drop=True)
        return pandas_df
    expected_exception = None
    if 'anon' in storage_options_extra:
        expected_exception = PermissionError('Forbidden')
    eval_general(pd, pandas, lambda module, **kwargs: pd.read_csv_glob(s3_path, **kwargs) if hasattr(module, 'read_csv_glob') else _pandas_read_csv_glob(s3_path, **kwargs), storage_options=s3_storage_options | storage_options_extra, expected_exception=expected_exception)