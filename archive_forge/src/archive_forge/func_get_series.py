from datetime import (
import re
import numpy as np
import pytest
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.reshape.concat import concat
from pandas.core.reshape.merge import (
def get_series():
    return [Series([1], dtype='int64'), Series([1], dtype='Int64'), Series([1.23]), Series(['foo']), Series([True]), Series([pd.Timestamp('2018-01-01')]), Series([pd.Timestamp('2018-01-01', tz='US/Eastern')])]