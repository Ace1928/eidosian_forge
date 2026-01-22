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
def exp_parts_cats_col(self):
    cats3 = Categorical(['a', 'a', 'b', 'b', 'a', 'a', 'a'], categories=['a', 'b'])
    idx3 = Index(['h', 'i', 'j', 'k', 'l', 'm', 'n'])
    values3 = [1, 1, 1, 1, 1, 1, 1]
    exp_parts_cats_col = DataFrame({'cats': cats3, 'values': values3}, index=idx3)
    return exp_parts_cats_col