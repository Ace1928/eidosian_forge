import numpy as np
import pytest
from pandas import (
from pandas.io.formats.style import Styler
from pandas.io.formats.style_render import _str_escape
@pytest.mark.parametrize('chars, expected', [('$ \\$&%#_{}~^\\ $ &%#_{}~^\\ $', ''.join(['$ \\$&%#_{}~^\\ $ ', '\\&\\%\\#\\_\\{\\}\\textasciitilde \\textasciicircum ', '\\textbackslash \\space \\$'])), ('\\( &%#_{}~^\\ \\) &%#_{}~^\\ \\(', ''.join(['\\( &%#_{}~^\\ \\) ', '\\&\\%\\#\\_\\{\\}\\textasciitilde \\textasciicircum ', '\\textbackslash \\space \\textbackslash ('])), ('$\\&%#_{}^\\$', '\\$\\textbackslash \\&\\%\\#\\_\\{\\}\\textasciicircum \\textbackslash \\$'), ('$ \\frac{1}{2} $ \\( \\frac{1}{2} \\)', ''.join(['$ \\frac{1}{2} $', ' \\textbackslash ( \\textbackslash frac\\{1\\}\\{2\\} \\textbackslash )']))])
def test_format_escape_latex_math(chars, expected):
    df = DataFrame([[chars]])
    s = df.style.format('{0}', escape='latex-math')
    assert s._translate(True, True)['body'][0][1]['display_value'] == expected