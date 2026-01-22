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
def exp_multi_row(self):
    cats2 = Categorical(['a', 'a', 'b', 'b', 'a', 'a', 'a'], categories=['a', 'b'])
    idx2 = Index(['h', 'i', 'j', 'k', 'l', 'm', 'n'])
    values2 = [1, 1, 2, 2, 1, 1, 1]
    exp_multi_row = DataFrame({'cats': cats2, 'values': values2}, index=idx2)
    return exp_multi_row