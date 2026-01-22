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
@pytest.mark.parametrize('decimals, expected_results', [(0, [1.0, 2.0, 3.0]), (1, [1.2, 2.3, 3.5]), (2, [1.23, 2.35, 3.46])])
def test_index_round(self, decimals, expected_results):
    idx = Index([1.234, 2.345, 3.456])
    result = idx.round(decimals)
    expected = Index(expected_results)
    tm.assert_index_equal(result, expected)