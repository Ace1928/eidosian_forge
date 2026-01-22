from collections import defaultdict
from datetime import datetime
from functools import partial
import math
import operator
import re
import numpy as np
import pytest
from pandas.compat import IS64
from pandas.errors import InvalidIndexError
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.indexes.api import (
def test_asof_numeric_vs_bool_raises(self):
    left = Index([1, 2, 3])
    right = Index([True, False], dtype=object)
    msg = 'Cannot compare dtypes int64 and bool'
    with pytest.raises(TypeError, match=msg):
        left.asof(right[0])
    with pytest.raises(InvalidIndexError, match=re.escape(str(right))):
        left.asof(right)
    with pytest.raises(InvalidIndexError, match=re.escape(str(left))):
        right.asof(left)