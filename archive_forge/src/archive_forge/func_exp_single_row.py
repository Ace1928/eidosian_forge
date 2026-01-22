from collections import namedtuple
from datetime import (
from decimal import Decimal
import re
import numpy as np
import pytest
from pandas._libs import iNaT
from pandas.errors import (
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import is_integer
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.fixture
def exp_single_row(self):
    cats1 = Categorical(['a', 'a', 'b', 'a', 'a', 'a', 'a'], categories=['a', 'b'])
    idx1 = Index(['h', 'i', 'j', 'k', 'l', 'm', 'n'])
    values1 = [1, 1, 2, 1, 1, 1, 1]
    exp_single_row = DataFrame({'cats': cats1, 'values': values1}, index=idx1)
    return exp_single_row