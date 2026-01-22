from io import StringIO
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_index_col_is_true(all_parsers):
    data = 'a,b\n1,2'
    parser = all_parsers
    msg = "The value of index_col couldn't be 'True'"
    with pytest.raises(ValueError, match=msg):
        parser.read_csv(StringIO(data), index_col=True)