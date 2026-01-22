import io
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.util import _test_decorators as td
@pytest.mark.single_cpu
@td.skip_array_manager_not_yet_implemented
def test_s3_parquet(s3_public_bucket, s3so, df1):
    pytest.importorskip('fastparquet')
    pytest.importorskip('s3fs')
    fn = f's3://{s3_public_bucket.name}/test.parquet'
    df1.to_parquet(fn, index=False, engine='fastparquet', compression=None, storage_options=s3so)
    df2 = read_parquet(fn, engine='fastparquet', storage_options=s3so)
    tm.assert_equal(df1, df2)