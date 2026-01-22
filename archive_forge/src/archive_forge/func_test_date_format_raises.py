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
def test_date_format_raises(self, df_table):
    msg = "Trying to write with `orient='table'` and `date_format='epoch'`. Table Schema requires dates to be formatted with `date_format='iso'`"
    with pytest.raises(ValueError, match=msg):
        df_table.to_json(orient='table', date_format='epoch')
    df_table.to_json(orient='table', date_format='iso')
    df_table.to_json(orient='table')