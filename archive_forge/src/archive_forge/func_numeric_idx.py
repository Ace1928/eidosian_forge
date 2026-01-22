from __future__ import annotations
from collections import abc
from datetime import timedelta
from decimal import Decimal
import operator
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import ops
from pandas.core.computation import expressions as expr
from pandas.tests.arithmetic.common import (
@pytest.fixture(params=[Index(np.arange(5, dtype='float64')), Index(np.arange(5, dtype='int64')), Index(np.arange(5, dtype='uint64')), RangeIndex(5)], ids=lambda x: type(x).__name__)
def numeric_idx(request):
    """
    Several types of numeric-dtypes Index objects
    """
    return request.param