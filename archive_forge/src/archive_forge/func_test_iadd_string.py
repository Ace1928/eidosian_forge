import datetime
from decimal import Decimal
import operator
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import ops
def test_iadd_string(self):
    index = pd.Index(['a', 'b', 'c'])
    assert 'a' in index
    index += '_x'
    assert 'a_x' in index