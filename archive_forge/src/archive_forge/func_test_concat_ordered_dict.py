from collections import (
from collections.abc import Iterator
from datetime import datetime
from decimal import Decimal
import numpy as np
import pytest
from pandas.errors import InvalidIndexError
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import SparseArray
from pandas.tests.extension.decimal import to_decimal
def test_concat_ordered_dict(self):
    expected = concat([Series(range(3)), Series(range(4))], keys=['First', 'Another'])
    result = concat({'First': Series(range(3)), 'Another': Series(range(4))})
    tm.assert_series_equal(result, expected)