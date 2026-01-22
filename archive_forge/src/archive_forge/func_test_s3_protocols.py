import io
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.util import _test_decorators as td
@pytest.mark.single_cpu
@pytest.mark.parametrize('protocol', ['s3', 's3a', 's3n'])
def test_s3_protocols(s3_public_bucket_with_data, tips_file, protocol, s3so):
    pytest.importorskip('s3fs')
    tm.assert_equal(read_csv(f'{protocol}://{s3_public_bucket_with_data.name}/tips.csv', storage_options=s3so), read_csv(tips_file))