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
@pytest.mark.parametrize('date_dtype', [np.dtype('<M8[ns]'), PeriodDtype('D'), DatetimeTZDtype('ns', 'US/Central')])
def test_as_json_table_type_date_dtypes(self, date_dtype):
    assert as_json_table_type(date_dtype) == 'datetime'