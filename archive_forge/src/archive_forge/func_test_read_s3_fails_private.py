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
def test_read_s3_fails_private(self, s3_private_bucket, s3so):
    msg = 'The specified bucket does not exist'
    with pytest.raises(OSError, match=msg):
        read_csv(f's3://{s3_private_bucket.name}/file.csv')