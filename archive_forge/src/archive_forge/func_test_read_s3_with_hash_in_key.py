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
def test_read_s3_with_hash_in_key(self, s3_public_bucket_with_data, tips_df, s3so):
    result = read_csv(f's3://{s3_public_bucket_with_data.name}/tips#1.csv', storage_options=s3so)
    tm.assert_frame_equal(tips_df, result)