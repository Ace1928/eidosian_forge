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
def test_cell_magic(ipython_with_magic, clean_globalenv):
    ipython_with_magic.push({'x': np.arange(5), 'y': np.array([3, 5, 4, 6, 7])})
    snippet = textwrap.dedent("\n    print(summary(a))\n    plot(x, y, pch=23, bg='orange', cex=2)\n    plot(x, x)\n    print(summary(x))\n    r = resid(a)\n    xc = coef(a)\n    ")
    ipython_with_magic.run_cell_magic('R', '-i x,y -o r,xc -w 150 -u mm a=lm(y~x)', snippet)
    np.testing.assert_almost_equal(ipython_with_magic.user_ns['xc'], [3.2, 0.9])
    np.testing.assert_almost_equal(ipython_with_magic.user_ns['r'], np.array([-0.2, 0.9, -1.0, 0.1, 0.2]))