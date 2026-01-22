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
def test_constructor_scalar_inference(self, using_infer_string):
    data = {'int': 1, 'bool': True, 'float': 3.0, 'complex': 4j, 'object': 'foo'}
    df = DataFrame(data, index=np.arange(10))
    assert df['int'].dtype == np.int64
    assert df['bool'].dtype == np.bool_
    assert df['float'].dtype == np.float64
    assert df['complex'].dtype == np.complex128
    assert df['object'].dtype == np.object_ if not using_infer_string else 'string'