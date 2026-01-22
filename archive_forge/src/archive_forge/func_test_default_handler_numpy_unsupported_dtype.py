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
def test_default_handler_numpy_unsupported_dtype(self):
    df = DataFrame({'a': [1, 2.3, complex(4, -5)], 'b': [float('nan'), None, complex(1.2, 0)]}, columns=['a', 'b'])
    expected = '[["(1+0j)","(nan+0j)"],["(2.3+0j)","(nan+0j)"],["(4-5j)","(1.2+0j)"]]'
    assert df.to_json(default_handler=str, orient='values') == expected