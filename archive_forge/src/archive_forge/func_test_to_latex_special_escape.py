import codecs
from datetime import datetime
from textwrap import dedent
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_to_latex_special_escape(self):
    df = DataFrame(['a\\b\\c', '^a^b^c', '~a~b~c'])
    result = df.to_latex(escape=True)
    expected = _dedent('\n            \\begin{tabular}{ll}\n            \\toprule\n             & 0 \\\\\n            \\midrule\n            0 & a\\textbackslash b\\textbackslash c \\\\\n            1 & \\textasciicircum a\\textasciicircum b\\textasciicircum c \\\\\n            2 & \\textasciitilde a\\textasciitilde b\\textasciitilde c \\\\\n            \\bottomrule\n            \\end{tabular}\n            ')
    assert result == expected