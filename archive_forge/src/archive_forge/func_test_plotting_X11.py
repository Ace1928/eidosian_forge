import pytest
import textwrap
import types
import warnings
from itertools import product
import rpy2.rinterface_lib.callbacks
import rpy2.rinterface_lib._rinterface_capi
import rpy2.robjects
import rpy2.robjects.conversion
from .. import utils
from io import StringIO
from rpy2 import rinterface
from rpy2.robjects import r, vectors, globalenv
import rpy2.robjects.packages as rpacks
@pytest.mark.skipif(IPython is None, reason='The optional package IPython cannot be imported.')
@pytest.mark.skipif(not has_numpy, reason='numpy not installed')
@pytest.mark.skip(reason='Test for X11 skipped.')
def test_plotting_X11(ipython_with_magic, clean_globalenv):
    ipython_with_magic.push({'x': np.arange(5), 'y': np.array([3, 5, 4, 6, 7])})
    cell = textwrap.dedent("\n    plot(x, y, pch=23, bg='orange', cex=2)\n    ")
    ipython_with_magic.run_line_magic('Rdevice', 'X11')
    ipython_with_magic.run_cell_magic('R', '', cell)