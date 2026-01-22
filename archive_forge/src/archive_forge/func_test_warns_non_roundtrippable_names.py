from collections import OrderedDict
from io import StringIO
import json
import numpy as np
import pytest
from pandas.core.dtypes.dtypes import (
import pandas as pd
from pandas import DataFrame
import pandas._testing as tm
from pandas.io.json._table_schema import (
@pytest.mark.parametrize('idx', [pd.Index([], name='index'), pd.MultiIndex.from_arrays([['foo'], ['bar']], names=('level_0', 'level_1')), pd.MultiIndex.from_arrays([['foo'], ['bar']], names=('foo', 'level_1'))])
def test_warns_non_roundtrippable_names(self, idx):
    df = DataFrame(index=idx)
    df.index.name = 'index'
    with tm.assert_produces_warning():
        set_default_names(df)