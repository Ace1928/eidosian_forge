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
def test_dataframe_div_silenced():
    pdf1 = pd.DataFrame({'A': np.arange(10), 'B': [np.nan, 1, 2, 3, 4] * 2, 'C': [np.nan] * 10, 'D': np.arange(10)}, index=list('abcdefghij'), columns=list('ABCD'))
    pdf2 = pd.DataFrame(np.random.default_rng(2).standard_normal((10, 4)), index=list('abcdefghjk'), columns=list('ABCX'))
    with tm.assert_produces_warning(None):
        pdf1.div(pdf2, fill_value=0)