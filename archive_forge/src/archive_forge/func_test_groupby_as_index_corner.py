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
def test_groupby_as_index_corner(df, ts):
    msg = 'as_index=False only valid with DataFrame'
    with pytest.raises(TypeError, match=msg):
        ts.groupby(lambda x: x.weekday(), as_index=False)
    msg = 'as_index=False only valid for axis=0'
    depr_msg = 'DataFrame.groupby with axis=1 is deprecated'
    with pytest.raises(ValueError, match=msg):
        with tm.assert_produces_warning(FutureWarning, match=depr_msg):
            df.groupby(lambda x: x.lower(), as_index=False, axis=1)