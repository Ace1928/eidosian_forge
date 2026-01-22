from textwrap import dedent
import numpy as np
import pytest
from pandas import (
from pandas.io.formats.style import Styler
from pandas.io.formats.style_render import (
def test_hidden_index(styler):
    styler.hide(axis='index')
    expected = dedent('        \\begin{tabular}{rrl}\n        A & B & C \\\\\n        0 & -0.61 & ab \\\\\n        1 & -1.22 & cd \\\\\n        \\end{tabular}\n        ')
    assert styler.to_latex() == expected