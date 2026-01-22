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
@pytest.mark.parametrize('dtype, missing, skipna, expected', [(float, np.nan, False, 'floating'), (float, np.nan, True, 'floating'), (object, np.nan, False, 'floating'), (object, np.nan, True, 'empty'), (object, None, False, 'mixed'), (object, None, True, 'empty')])
@pytest.mark.parametrize('box', [Series, np.array])
def test_object_empty(self, box, missing, dtype, skipna, expected):
    arr = box([missing, missing], dtype=dtype)
    result = lib.infer_dtype(arr, skipna=skipna)
    assert result == expected