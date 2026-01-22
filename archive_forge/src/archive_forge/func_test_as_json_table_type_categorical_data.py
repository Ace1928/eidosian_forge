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
@pytest.mark.parametrize('cat_data', [pd.Categorical(['a']), pd.Categorical([1]), pd.Series(pd.Categorical([1])), pd.CategoricalIndex([1]), pd.Categorical([1])])
def test_as_json_table_type_categorical_data(self, cat_data):
    assert as_json_table_type(cat_data.dtype) == 'any'