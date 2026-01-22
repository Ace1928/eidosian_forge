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
def test_constructor_list_of_namedtuples(self):
    named_tuple = namedtuple('Pandas', list('ab'))
    tuples = [named_tuple(1, 3), named_tuple(2, 4)]
    expected = DataFrame({'a': [1, 2], 'b': [3, 4]})
    result = DataFrame(tuples)
    tm.assert_frame_equal(result, expected)
    expected = DataFrame({'y': [1, 2], 'z': [3, 4]})
    result = DataFrame(tuples, columns=['y', 'z'])
    tm.assert_frame_equal(result, expected)