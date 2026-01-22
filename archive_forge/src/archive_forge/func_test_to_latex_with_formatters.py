import codecs
from datetime import datetime
from textwrap import dedent
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_to_latex_with_formatters(self):
    df = DataFrame({'datetime64': [datetime(2016, 1, 1), datetime(2016, 2, 5), datetime(2016, 3, 3)], 'float': [1.0, 2.0, 3.0], 'int': [1, 2, 3], 'object': [(1, 2), True, False]})
    formatters = {'datetime64': lambda x: x.strftime('%Y-%m'), 'float': lambda x: f'[{x: 4.1f}]', 'int': lambda x: f'0x{x:x}', 'object': lambda x: f'-{x!s}-', '__index__': lambda x: f'index: {x}'}
    result = df.to_latex(formatters=dict(formatters))
    expected = _dedent('\n            \\begin{tabular}{llrrl}\n            \\toprule\n             & datetime64 & float & int & object \\\\\n            \\midrule\n            index: 0 & 2016-01 & [ 1.0] & 0x1 & -(1, 2)- \\\\\n            index: 1 & 2016-02 & [ 2.0] & 0x2 & -True- \\\\\n            index: 2 & 2016-03 & [ 3.0] & 0x3 & -False- \\\\\n            \\bottomrule\n            \\end{tabular}\n            ')
    assert result == expected