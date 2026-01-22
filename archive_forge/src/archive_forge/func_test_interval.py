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
@pytest.mark.parametrize('asobject', [True, False])
def test_interval(self, asobject):
    idx = pd.IntervalIndex.from_breaks(range(5), closed='both')
    if asobject:
        idx = idx.astype(object)
    inferred = lib.infer_dtype(idx, skipna=False)
    assert inferred == 'interval'
    inferred = lib.infer_dtype(idx._data, skipna=False)
    assert inferred == 'interval'
    inferred = lib.infer_dtype(Series(idx, dtype=idx.dtype), skipna=False)
    assert inferred == 'interval'