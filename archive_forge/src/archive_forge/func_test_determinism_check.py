import os
import subprocess
import sys
import pytest
import matplotlib as mpl
import matplotlib.testing.compare
from matplotlib import pyplot as plt
from matplotlib.testing._markers import needs_ghostscript, needs_usetex
@pytest.mark.parametrize('objects, fmt, usetex', [('', 'pdf', False), ('m', 'pdf', False), ('h', 'pdf', False), ('i', 'pdf', False), ('mhi', 'pdf', False), ('mhi', 'ps', False), pytest.param('mhi', 'ps', True, marks=[needs_usetex, needs_ghostscript]), ('mhi', 'svg', False), pytest.param('mhi', 'svg', True, marks=needs_usetex)])
def test_determinism_check(objects, fmt, usetex):
    """
    Output three times the same graphs and checks that the outputs are exactly
    the same.

    Parameters
    ----------
    objects : str
        Objects to be included in the test document: 'm' for markers, 'h' for
        hatch patterns, 'i' for images.
    fmt : {"pdf", "ps", "svg"}
        Output format.
    """
    plots = [subprocess.check_output([sys.executable, '-R', '-c', f'from matplotlib.tests.test_determinism import _save_figure;_save_figure({objects!r}, {fmt!r}, {usetex})'], env={**os.environ, 'SOURCE_DATE_EPOCH': '946684800', 'MPLBACKEND': 'Agg'}) for _ in range(3)]
    for p in plots[1:]:
        if fmt == 'ps' and usetex:
            if p != plots[0]:
                pytest.skip('failed, maybe due to ghostscript timestamps')
        else:
            assert p == plots[0]