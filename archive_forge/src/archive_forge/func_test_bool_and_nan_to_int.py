from io import StringIO
import numpy as np
import pytest
from pandas._libs.parsers import STR_NA_VALUES
from pandas import (
import pandas._testing as tm
def test_bool_and_nan_to_int(all_parsers):
    parser = all_parsers
    data = '0\nNaN\nTrue\nFalse\n'
    with pytest.raises(ValueError, match='convert|NoneType'):
        parser.read_csv(StringIO(data), dtype='int')