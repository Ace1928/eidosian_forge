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
@pytest.mark.parametrize('constructor', [lambda: DataFrame(), lambda: DataFrame(None), lambda: DataFrame(()), lambda: DataFrame([]), lambda: DataFrame((_ for _ in [])), lambda: DataFrame(range(0)), lambda: DataFrame(data=None), lambda: DataFrame(data=()), lambda: DataFrame(data=[]), lambda: DataFrame(data=(_ for _ in [])), lambda: DataFrame(data=range(0))])
def test_empty_constructor(self, constructor):
    expected = DataFrame()
    result = constructor()
    assert len(result.index) == 0
    assert len(result.columns) == 0
    tm.assert_frame_equal(result, expected)