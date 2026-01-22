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
@pytest.mark.skipif(has_pandas or not has_numpy, reason='numpy not installed')
def test_Rconverter_numpy(ipython_with_magic, clean_globalenv):
    dataf_np = np.array([(1, 2.9, 'a'), (2, 3.5, 'b'), (3, 2.1, 'c')], dtype=[('x', '<i4'), ('y', '<f8'), ('z', '|%s1' % np_string_type)])
    _test_Rconverter(ipython_with_magic, clean_globalenv, dataf_np, np.ndarray)