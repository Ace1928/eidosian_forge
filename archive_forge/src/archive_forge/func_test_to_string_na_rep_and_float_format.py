from datetime import (
from io import StringIO
import re
import sys
from textwrap import dedent
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('na_rep', ['NaN', 'Ted'])
def test_to_string_na_rep_and_float_format(self, na_rep):
    df = DataFrame([['A', 1.2225], ['A', None]], columns=['Group', 'Data'])
    result = df.to_string(na_rep=na_rep, float_format='{:.2f}'.format)
    expected = dedent(f'               Group  Data\n             0     A  1.22\n             1     A   {na_rep}')
    assert result == expected