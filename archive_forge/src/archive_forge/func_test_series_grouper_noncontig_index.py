from datetime import datetime
import decimal
from decimal import Decimal
import re
import numpy as np
import pytest
from pandas.errors import (
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import is_string_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import BooleanArray
import pandas.core.common as com
def test_series_grouper_noncontig_index():
    index = Index(['a' * 10] * 100)
    values = Series(np.random.default_rng(2).standard_normal(50), index=index[::2])
    labels = np.random.default_rng(2).integers(0, 5, 50)
    grouped = values.groupby(labels)
    f = lambda x: len(set(map(id, x.index)))
    grouped.agg(f)