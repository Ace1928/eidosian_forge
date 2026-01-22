from datetime import datetime
import decimal
from decimal import Decimal
import re
import numpy as np
import pytest
from pandas.errors import (
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import is_string_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import BooleanArray
import pandas.core.common as com
def test_groupby_empty_list_raises():
    values = zip(range(10), range(10))
    df = DataFrame(values, columns=['apple', 'b'])
    msg = 'Grouper and axis must be same length'
    with pytest.raises(ValueError, match=msg):
        df.groupby([[]])