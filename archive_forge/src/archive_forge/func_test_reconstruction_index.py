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
def test_reconstruction_index(self):
    df = DataFrame([[1, 2, 3], [4, 5, 6]])
    result = read_json(StringIO(df.to_json()))
    tm.assert_frame_equal(result, df)
    df = DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]}, index=['A', 'B', 'C'])
    result = read_json(StringIO(df.to_json()))
    tm.assert_frame_equal(result, df)