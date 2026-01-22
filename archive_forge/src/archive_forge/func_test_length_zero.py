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
@pytest.mark.parametrize('skipna', [True, False])
def test_length_zero(self, skipna):
    result = lib.infer_dtype(np.array([], dtype='i4'), skipna=skipna)
    assert result == 'integer'
    result = lib.infer_dtype([], skipna=skipna)
    assert result == 'empty'
    arr = np.array([np.array([], dtype=object), np.array([], dtype=object)])
    result = lib.infer_dtype(arr, skipna=skipna)
    assert result == 'empty'