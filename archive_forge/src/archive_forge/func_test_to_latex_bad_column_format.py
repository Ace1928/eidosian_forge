import codecs
from datetime import datetime
from textwrap import dedent
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('bad_column_format', [5, 1.2, ['l', 'r'], ('r', 'c'), {'r', 'c', 'l'}, {'a': 'r', 'b': 'l'}])
def test_to_latex_bad_column_format(self, bad_column_format):
    df = DataFrame({'a': [1, 2], 'b': ['b1', 'b2']})
    msg = '`column_format` must be str or unicode'
    with pytest.raises(ValueError, match=msg):
        df.to_latex(column_format=bad_column_format)