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
def test_constructor_dict_of_tuples(self):
    data = {'a': (1, 2, 3), 'b': (4, 5, 6)}
    result = DataFrame(data)
    expected = DataFrame({k: list(v) for k, v in data.items()})
    tm.assert_frame_equal(result, expected, check_dtype=False)