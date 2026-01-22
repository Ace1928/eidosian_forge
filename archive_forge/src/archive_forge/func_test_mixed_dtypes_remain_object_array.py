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
def test_mixed_dtypes_remain_object_array(self):
    arr = np.array([datetime(2015, 1, 1, tzinfo=pytz.utc), 1], dtype=object)
    result = lib.maybe_convert_objects(arr, convert_non_numeric=True)
    tm.assert_numpy_array_equal(result, arr)