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
def test_constructor_mixed_dict_and_Series(self):
    data = {}
    data['A'] = {'foo': 1, 'bar': 2, 'baz': 3}
    data['B'] = Series([4, 3, 2, 1], index=['bar', 'qux', 'baz', 'foo'])
    result = DataFrame(data)
    assert result.index.is_monotonic_increasing
    with pytest.raises(ValueError, match='ambiguous ordering'):
        DataFrame({'A': ['a', 'b'], 'B': {'a': 'a', 'b': 'b'}})
    result = DataFrame({'A': ['a', 'b'], 'B': Series(['a', 'b'], index=['a', 'b'])})
    expected = DataFrame({'A': ['a', 'b'], 'B': ['a', 'b']}, index=['a', 'b'])
    tm.assert_frame_equal(result, expected)