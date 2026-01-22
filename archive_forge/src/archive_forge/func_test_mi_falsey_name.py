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
def test_mi_falsey_name(self):
    df = DataFrame(np.random.default_rng(2).standard_normal((4, 4)), index=pd.MultiIndex.from_product([('A', 'B'), ('a', 'b')]))
    result = [x['name'] for x in build_table_schema(df)['fields']]
    assert result == ['level_0', 'level_1', 0, 1, 2, 3]