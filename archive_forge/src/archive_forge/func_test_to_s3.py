import datetime
from datetime import timedelta
from decimal import Decimal
from io import (
import json
import os
import sys
import time
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas.compat import IS64
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
from pandas.core.arrays.string_arrow import ArrowStringArrayNumpySemantics
from pandas.io.json import ujson_dumps
@pytest.mark.single_cpu
def test_to_s3(self, s3_public_bucket, s3so):
    mock_bucket_name, target_file = (s3_public_bucket.name, 'test.json')
    df = DataFrame({'x': [1, 2, 3], 'y': [2, 4, 6]})
    df.to_json(f's3://{mock_bucket_name}/{target_file}', storage_options=s3so)
    timeout = 5
    while True:
        if target_file in (obj.key for obj in s3_public_bucket.objects.all()):
            break
        time.sleep(0.1)
        timeout -= 0.1
        assert timeout > 0, 'Timed out waiting for file to appear on moto'