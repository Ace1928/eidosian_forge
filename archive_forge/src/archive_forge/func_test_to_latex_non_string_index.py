import codecs
from datetime import datetime
from textwrap import dedent
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_to_latex_non_string_index(self):
    df = DataFrame([[1, 2, 3]] * 2).set_index([0, 1])
    result = df.to_latex(multirow=False)
    expected = _dedent('\n            \\begin{tabular}{llr}\n            \\toprule\n             &  & 2 \\\\\n            0 & 1 &  \\\\\n            \\midrule\n            1 & 2 & 3 \\\\\n             & 2 & 3 \\\\\n            \\bottomrule\n            \\end{tabular}\n            ')
    assert result == expected