import codecs
from datetime import datetime
from textwrap import dedent
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('na_rep', ['NaN', 'Ted'])
def test_to_latex_na_rep_and_float_format(self, na_rep):
    df = DataFrame([['A', 1.2225], ['A', None]], columns=['Group', 'Data'])
    result = df.to_latex(na_rep=na_rep, float_format='{:.2f}'.format)
    expected = _dedent(f'\n            \\begin{{tabular}}{{llr}}\n            \\toprule\n             & Group & Data \\\\\n            \\midrule\n            0 & A & 1.22 \\\\\n            1 & A & {na_rep} \\\\\n            \\bottomrule\n            \\end{{tabular}}\n            ')
    assert result == expected