from io import StringIO
import numpy as np
import pytest
from pandas.errors import ParserError
from pandas import (
import pandas._testing as tm
def test_raise_on_mixed_dtype_usecols(all_parsers):
    data = 'a,b,c\n        1000,2000,3000\n        4000,5000,6000\n        '
    usecols = [0, 'b', 2]
    parser = all_parsers
    with pytest.raises(ValueError, match=_msg_validate_usecols_arg):
        parser.read_csv(StringIO(data), usecols=usecols)