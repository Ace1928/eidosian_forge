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
def test_nested_dict_construction(self):
    columns = ['Nevada', 'Ohio']
    pop = {'Nevada': {2001: 2.4, 2002: 2.9}, 'Ohio': {2000: 1.5, 2001: 1.7, 2002: 3.6}}
    result = DataFrame(pop, index=[2001, 2002, 2003], columns=columns)
    expected = DataFrame([(2.4, 1.7), (2.9, 3.6), (np.nan, np.nan)], columns=columns, index=Index([2001, 2002, 2003]))
    tm.assert_frame_equal(result, expected)