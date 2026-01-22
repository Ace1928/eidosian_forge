import array
from collections import (
from collections.abc import Iterator
from dataclasses import make_dataclass
from datetime import (
import functools
import re
import numpy as np
from numpy import ma
from numpy.ma import mrecords
import pytest
import pytz
from pandas._config import using_pyarrow_string_dtype
from pandas._libs import lib
from pandas.errors import IntCastingNaNError
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import is_integer_dtype
from pandas.core.dtypes.dtypes import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.arrays import (
@pytest.mark.parametrize('klass,name', [(lambda x: np.timedelta64(x, 'D'), 'timedelta64'), (lambda x: timedelta(days=x), 'pytimedelta'), (lambda x: Timedelta(x, 'D'), 'Timedelta[ns]'), (lambda x: Timedelta(x, 'D').as_unit('s'), 'Timedelta[s]')])
def test_constructor_dict_timedelta64_index(self, klass, name):
    td_as_int = [1, 2, 3, 4]
    data = {i: {klass(s): 2 * i} for i, s in enumerate(td_as_int)}
    expected = DataFrame([{0: 0, 1: None, 2: None, 3: None}, {0: None, 1: 2, 2: None, 3: None}, {0: None, 1: None, 2: 4, 3: None}, {0: None, 1: None, 2: None, 3: 6}], index=[Timedelta(td, 'D') for td in td_as_int])
    result = DataFrame(data)
    tm.assert_frame_equal(result, expected)