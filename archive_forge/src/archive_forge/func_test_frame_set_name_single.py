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
def test_frame_set_name_single(df):
    grouped = df.groupby('A')
    result = grouped.mean(numeric_only=True)
    assert result.index.name == 'A'
    result = df.groupby('A', as_index=False).mean(numeric_only=True)
    assert result.index.name != 'A'
    result = grouped[['C', 'D']].agg('mean')
    assert result.index.name == 'A'
    result = grouped.agg({'C': 'mean', 'D': 'std'})
    assert result.index.name == 'A'
    result = grouped['C'].mean()
    assert result.index.name == 'A'
    result = grouped['C'].agg('mean')
    assert result.index.name == 'A'
    result = grouped['C'].agg(['mean', 'std'])
    assert result.index.name == 'A'
    msg = 'nested renamer is not supported'
    with pytest.raises(SpecificationError, match=msg):
        grouped['C'].agg({'foo': 'mean', 'bar': 'std'})