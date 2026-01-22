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
@pytest.mark.parametrize('bool_dtype', [bool, np.bool_])
def test_as_json_table_type_bool_dtypes(self, bool_dtype):
    assert as_json_table_type(bool_dtype) == 'boolean'