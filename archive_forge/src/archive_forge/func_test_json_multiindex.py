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
@pytest.mark.parametrize('dataframe,expected', [(DataFrame({'x': [1, 2, 3], 'y': ['a', 'b', 'c']}), '{"(0, \'x\')":1,"(0, \'y\')":"a","(1, \'x\')":2,"(1, \'y\')":"b","(2, \'x\')":3,"(2, \'y\')":"c"}')])
def test_json_multiindex(self, dataframe, expected):
    series = dataframe.stack(future_stack=True)
    result = series.to_json(orient='index')
    assert result == expected