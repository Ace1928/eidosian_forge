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
def test_groupby_level_apply(multiindex_dataframe_random_data):
    result = multiindex_dataframe_random_data.groupby(level=0).count()
    assert result.index.name == 'first'
    result = multiindex_dataframe_random_data.groupby(level=1).count()
    assert result.index.name == 'second'
    result = multiindex_dataframe_random_data['A'].groupby(level=0).count()
    assert result.index.name == 'first'