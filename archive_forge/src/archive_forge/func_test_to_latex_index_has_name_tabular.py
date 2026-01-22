import codecs
from datetime import datetime
from textwrap import dedent
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_to_latex_index_has_name_tabular(self):
    df = DataFrame({'a': [0, 0, 1, 1], 'b': list('abab'), 'c': [1, 2, 3, 4]})
    result = df.set_index(['a', 'b']).to_latex(multirow=False)
    expected = _dedent('\n            \\begin{tabular}{llr}\n            \\toprule\n             &  & c \\\\\n            a & b &  \\\\\n            \\midrule\n            0 & a & 1 \\\\\n             & b & 2 \\\\\n            1 & a & 3 \\\\\n             & b & 4 \\\\\n            \\bottomrule\n            \\end{tabular}\n            ')
    assert result == expected