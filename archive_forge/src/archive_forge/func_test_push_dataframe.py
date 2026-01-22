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
@pytest.mark.skipif(not has_pandas, reason='pandas is not available in python')
@pytest.mark.skipif(not has_numpy, reason='numpy not installed')
def test_push_dataframe(ipython_with_magic, clean_globalenv):
    df = pd.DataFrame([{'a': 1, 'b': 'bar'}, {'a': 5, 'b': 'foo', 'c': 20}])
    ipython_with_magic.push({'df': df})
    ipython_with_magic.run_line_magic('Rpush', 'df')
    sio = StringIO()
    with utils.obj_in_module(rpy2.rinterface_lib.callbacks, 'consolewrite_print', sio.write):
        r('print(df$b[1])')
        assert '[1] "bar"' in sio.getvalue()
    assert r('df$a[2]')[0] == 5
    missing = r('df$c[1]')[0]
    assert np.isnan(missing), missing