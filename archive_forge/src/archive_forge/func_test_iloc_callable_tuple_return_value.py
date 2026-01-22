from collections import namedtuple
from datetime import (
from decimal import Decimal
import re
import numpy as np
import pytest
from pandas._libs import iNaT
from pandas.errors import (
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import is_integer
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_iloc_callable_tuple_return_value(self):
    df = DataFrame(np.arange(40).reshape(10, 4), index=range(0, 20, 2))
    msg = 'callable with iloc'
    with tm.assert_produces_warning(FutureWarning, match=msg):
        df.iloc[lambda _: (0,)]
    with tm.assert_produces_warning(FutureWarning, match=msg):
        df.iloc[lambda _: (0,)] = 1