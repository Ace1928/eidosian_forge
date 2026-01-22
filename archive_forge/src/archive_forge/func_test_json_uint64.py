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
def test_json_uint64(self):
    expected = '{"columns":["col1"],"index":[0,1],"data":[[13342205958987758245],[12388075603347835679]]}'
    df = DataFrame(data={'col1': [13342205958987758245, 12388075603347835679]})
    result = df.to_json(orient='split')
    assert result == expected