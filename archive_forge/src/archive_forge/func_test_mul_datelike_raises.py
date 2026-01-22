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
def test_mul_datelike_raises(self, numeric_idx):
    idx = numeric_idx
    msg = 'cannot perform __rmul__ with this index type'
    with pytest.raises(TypeError, match=msg):
        idx * date_range('20130101', periods=5)