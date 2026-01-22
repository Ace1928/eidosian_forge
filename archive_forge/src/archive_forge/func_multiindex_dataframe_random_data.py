from __future__ import annotations
from collections import abc
from datetime import (
from decimal import Decimal
import operator
import os
from typing import (
from dateutil.tz import (
import hypothesis
from hypothesis import strategies as st
import numpy as np
import pytest
from pytz import (
from pandas._config.config import _get_option
import pandas.util._test_decorators as td
from pandas.core.dtypes.dtypes import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import ops
from pandas.core.indexes.api import (
from pandas.util.version import Version
import zoneinfo
@pytest.fixture
def multiindex_dataframe_random_data(lexsorted_two_level_string_multiindex) -> DataFrame:
    """DataFrame with 2 level MultiIndex with random data"""
    index = lexsorted_two_level_string_multiindex
    return DataFrame(np.random.default_rng(2).standard_normal((10, 3)), index=index, columns=Index(['A', 'B', 'C'], name='exp'))