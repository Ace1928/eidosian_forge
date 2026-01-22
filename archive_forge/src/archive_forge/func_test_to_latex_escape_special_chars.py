import codecs
from datetime import datetime
from textwrap import dedent
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_to_latex_escape_special_chars(self):
    special_characters = ['&', '%', '$', '#', '_', '{', '}', '~', '^', '\\']
    df = DataFrame(data=special_characters)
    result = df.to_latex(escape=True)
    expected = _dedent('\n            \\begin{tabular}{ll}\n            \\toprule\n             & 0 \\\\\n            \\midrule\n            0 & \\& \\\\\n            1 & \\% \\\\\n            2 & \\$ \\\\\n            3 & \\# \\\\\n            4 & \\_ \\\\\n            5 & \\{ \\\\\n            6 & \\} \\\\\n            7 & \\textasciitilde  \\\\\n            8 & \\textasciicircum  \\\\\n            9 & \\textbackslash  \\\\\n            \\bottomrule\n            \\end{tabular}\n            ')
    assert result == expected