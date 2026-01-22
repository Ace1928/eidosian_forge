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
def exp_single_cats_value(self):
    cats4 = Categorical(['a', 'a', 'b', 'a', 'a', 'a', 'a'], categories=['a', 'b'])
    idx4 = Index(['h', 'i', 'j', 'k', 'l', 'm', 'n'])
    values4 = [1, 1, 1, 1, 1, 1, 1]
    exp_single_cats_value = DataFrame({'cats': cats4, 'values': values4}, index=idx4)
    return exp_single_cats_value