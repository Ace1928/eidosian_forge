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
@pytest.mark.parametrize('date_data', [pd.to_datetime(['2016']), pd.to_datetime(['2016'], utc=True), pd.Series(pd.to_datetime(['2016'])), pd.Series(pd.to_datetime(['2016'], utc=True)), pd.period_range('2016', freq='Y', periods=3)])
def test_as_json_table_type_date_data(self, date_data):
    assert as_json_table_type(date_data.dtype) == 'datetime'