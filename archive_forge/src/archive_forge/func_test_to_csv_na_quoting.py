import csv
from io import StringIO
import os
import numpy as np
import pytest
from pandas.errors import ParserError
import pandas as pd
from pandas import (
import pandas._testing as tm
import pandas.core.common as com
from pandas.io.common import get_handle
def test_to_csv_na_quoting(self):
    result = DataFrame([None, None]).to_csv(None, header=False, index=False, na_rep='').replace('\r\n', '\n')
    expected = '""\n""\n'
    assert result == expected