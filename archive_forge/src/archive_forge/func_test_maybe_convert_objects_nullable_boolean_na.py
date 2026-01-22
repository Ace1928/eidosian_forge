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
@pytest.mark.parametrize('val', [None, np.nan])
def test_maybe_convert_objects_nullable_boolean_na(self, val):
    arr = np.array([True, False, val], dtype=object)
    exp = BooleanArray(np.array([True, False, False]), np.array([False, False, True]))
    out = lib.maybe_convert_objects(arr, convert_to_nullable_dtype=True)
    tm.assert_extension_array_equal(out, exp)