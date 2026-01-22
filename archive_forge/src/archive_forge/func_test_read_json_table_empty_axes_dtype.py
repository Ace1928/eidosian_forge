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
@pytest.mark.parametrize('orient', ['index', 'columns', 'records', 'values'])
def test_read_json_table_empty_axes_dtype(self, orient):
    expected = DataFrame()
    result = read_json(StringIO('{}'), orient=orient, convert_axes=True)
    tm.assert_index_equal(result.index, expected.index)
    tm.assert_index_equal(result.columns, expected.columns)