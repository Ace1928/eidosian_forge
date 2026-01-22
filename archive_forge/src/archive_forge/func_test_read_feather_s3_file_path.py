from io import BytesIO
import logging
import re
import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas import DataFrame
import pandas._testing as tm
from pandas.io.feather_format import read_feather
from pandas.io.parsers import read_csv
def test_read_feather_s3_file_path(self, s3_public_bucket_with_data, feather_file, s3so):
    pytest.importorskip('pyarrow')
    expected = read_feather(feather_file)
    res = read_feather(f's3://{s3_public_bucket_with_data.name}/simple_dataset.feather', storage_options=s3so)
    tm.assert_frame_equal(expected, res)