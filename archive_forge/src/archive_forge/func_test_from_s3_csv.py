import io
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.util import _test_decorators as td
@pytest.mark.single_cpu
def test_from_s3_csv(s3_public_bucket_with_data, tips_file, s3so):
    pytest.importorskip('s3fs')
    tm.assert_equal(read_csv(f's3://{s3_public_bucket_with_data.name}/tips.csv', storage_options=s3so), read_csv(tips_file))
    tm.assert_equal(read_csv(f's3://{s3_public_bucket_with_data.name}/tips.csv.gz', storage_options=s3so), read_csv(tips_file))
    tm.assert_equal(read_csv(f's3://{s3_public_bucket_with_data.name}/tips.csv.bz2', storage_options=s3so), read_csv(tips_file))