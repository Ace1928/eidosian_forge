import collections
from collections import namedtuple
from collections.abc import Iterator
from datetime import (
from decimal import Decimal
from fractions import Fraction
from io import StringIO
import itertools
from numbers import Number
import re
import sys
from typing import (
import numpy as np
import pytest
import pytz
from pandas._libs import (
from pandas.compat.numpy import np_version_gt2
from pandas.core.dtypes import inference
from pandas.core.dtypes.cast import find_result_type
from pandas.core.dtypes.common import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
@pytest.mark.parametrize('right,result', [(0, np.uint8), (-1, np.int16), (300, np.uint16), (300.0, np.uint16), (300.1, np.float64), (np.int16(300), np.int16 if np_version_gt2 else np.uint16)])
def test_find_result_type_uint_int(right, result):
    left_dtype = np.dtype('uint8')
    assert find_result_type(left_dtype, right) == result