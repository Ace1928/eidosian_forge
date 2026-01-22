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
@pytest.mark.parametrize('klass', [pd.array, Series])
@pytest.mark.parametrize('skipna', [True, False])
@pytest.mark.parametrize('data', [[True, False, True], [True, False, pd.NA]])
def test_boolean_dtype(self, data, skipna, klass):
    val = klass(data, dtype='boolean')
    inferred = lib.infer_dtype(val, skipna=skipna)
    assert inferred == 'boolean'