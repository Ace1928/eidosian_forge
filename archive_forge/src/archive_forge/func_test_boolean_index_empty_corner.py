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
def test_boolean_index_empty_corner(self):
    blah = DataFrame(np.empty([0, 1]), columns=['A'], index=DatetimeIndex([]))
    k = np.array([], bool)
    blah[k]
    blah[k] = 0