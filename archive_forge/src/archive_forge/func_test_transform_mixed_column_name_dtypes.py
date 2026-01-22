from itertools import chain
import re
import numpy as np
import pytest
from pandas.errors import SpecificationError
from pandas import (
import pandas._testing as tm
def test_transform_mixed_column_name_dtypes():
    df = DataFrame({'a': ['1']})
    msg = "Column\\(s\\) \\[1, 'b'\\] do not exist"
    with pytest.raises(KeyError, match=msg):
        df.transform({'a': int, 1: str, 'b': int})