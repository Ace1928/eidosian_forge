from collections import defaultdict
from io import StringIO
import numpy as np
import pytest
from pandas.errors import ParserWarning
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
def decimal_number_check(request, parser, numeric_decimal, thousands, float_precision):
    value = numeric_decimal[0]
    if thousands is None and value in ('1_,', '1_234,56', '1_234,56e0'):
        request.applymarker(pytest.mark.xfail(reason=f'thousands={thousands} and sep is in {value}'))
    df = parser.read_csv(StringIO(value), float_precision=float_precision, sep='|', thousands=thousands, decimal=',', header=None)
    val = df.iloc[0, 0]
    assert val == numeric_decimal[1]