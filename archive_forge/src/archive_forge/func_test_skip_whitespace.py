from collections import defaultdict
from io import StringIO
import numpy as np
import pytest
from pandas.errors import ParserWarning
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
@pytest.mark.parametrize('float_precision', [None, 'legacy', 'high', 'round_trip'])
def test_skip_whitespace(c_parser_only, float_precision):
    DATA = 'id\tnum\t\n1\t1.2 \t\n1\t 2.1\t\n2\t 1\t\n2\t 1.2 \t\n'
    df = c_parser_only.read_csv(StringIO(DATA), float_precision=float_precision, sep='\t', header=0, dtype={1: np.float64})
    tm.assert_series_equal(df.iloc[:, 1], pd.Series([1.2, 2.1, 1.0, 1.2], name='num'))