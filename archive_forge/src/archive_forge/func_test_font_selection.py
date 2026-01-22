import os
from pathlib import Path
import re
import subprocess
import sys
import matplotlib.pyplot as plt
from matplotlib.texmanager import TexManager
from matplotlib.testing._markers import needs_usetex
import pytest
@pytest.mark.parametrize('rc, preamble, family', [({'font.family': 'sans-serif', 'font.sans-serif': 'helvetica'}, '\\usepackage{helvet}', '\\sffamily'), ({'font.family': 'serif', 'font.serif': 'palatino'}, '\\usepackage{mathpazo}', '\\rmfamily'), ({'font.family': 'cursive', 'font.cursive': 'zapf chancery'}, '\\usepackage{chancery}', '\\rmfamily'), ({'font.family': 'monospace', 'font.monospace': 'courier'}, '\\usepackage{courier}', '\\ttfamily'), ({'font.family': 'helvetica'}, '\\usepackage{helvet}', '\\sffamily'), ({'font.family': 'palatino'}, '\\usepackage{mathpazo}', '\\rmfamily'), ({'font.family': 'zapf chancery'}, '\\usepackage{chancery}', '\\rmfamily'), ({'font.family': 'courier'}, '\\usepackage{courier}', '\\ttfamily')])
def test_font_selection(rc, preamble, family):
    plt.rcParams.update(rc)
    tm = TexManager()
    src = Path(tm.make_tex('hello, world', fontsize=12)).read_text()
    assert preamble in src
    assert [*re.findall('\\\\\\w+family', src)] == [family]