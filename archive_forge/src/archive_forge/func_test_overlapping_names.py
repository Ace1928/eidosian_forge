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
@pytest.mark.parametrize('case', [pd.Series([1], index=pd.Index([1], name='a'), name='a'), DataFrame({'A': [1]}, index=pd.Index([1], name='A')), DataFrame({'A': [1]}, index=pd.MultiIndex.from_arrays([['a'], [1]], names=['A', 'a']))])
def test_overlapping_names(self, case):
    with pytest.raises(ValueError, match='Overlapping'):
        case.to_json(orient='table')