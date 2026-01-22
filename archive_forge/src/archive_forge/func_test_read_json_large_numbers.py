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
@pytest.mark.parametrize('bigNum', [-2 ** 63 - 1, 2 ** 64])
def test_read_json_large_numbers(self, bigNum):
    json = StringIO('{"articleId":' + str(bigNum) + '}')
    msg = 'Value is too small|Value is too big'
    with pytest.raises(ValueError, match=msg):
        read_json(json)
    json = StringIO('{"0":{"articleId":' + str(bigNum) + '}}')
    with pytest.raises(ValueError, match=msg):
        read_json(json)