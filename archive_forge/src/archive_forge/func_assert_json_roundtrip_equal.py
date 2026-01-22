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
def assert_json_roundtrip_equal(result, expected, orient):
    if orient in ('records', 'values'):
        expected = expected.reset_index(drop=True)
    if orient == 'values':
        expected.columns = range(len(expected.columns))
    tm.assert_frame_equal(result, expected)