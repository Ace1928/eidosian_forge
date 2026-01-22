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
def test_push(ipython_with_magic, clean_globalenv):
    for obj in (np.arange(5), [1, 2]):
        ipython_with_magic.push({'X': obj})
        ipython_with_magic.run_line_magic('Rpush', 'X')
        np.testing.assert_almost_equal(np.asarray(r('X')), ipython_with_magic.user_ns['X'])