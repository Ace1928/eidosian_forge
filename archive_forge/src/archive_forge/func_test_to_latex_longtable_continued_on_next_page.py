import codecs
from datetime import datetime
from textwrap import dedent
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('df, expected_number', [(DataFrame({'a': [1, 2]}), 1), (DataFrame({'a': [1, 2], 'b': [3, 4]}), 2), (DataFrame({'a': [1, 2], 'b': [3, 4], 'c': [5, 6]}), 3)])
def test_to_latex_longtable_continued_on_next_page(self, df, expected_number):
    result = df.to_latex(index=False, longtable=True)
    assert f'\\multicolumn{{{expected_number}}}' in result