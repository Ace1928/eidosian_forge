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
def test_rmagic_localscope(ipython_with_magic, clean_globalenv):
    ipython_with_magic.push({'x': 0})
    ipython_with_magic.run_line_magic('R', '-i x -o result result <-x+1')
    result = ipython_with_magic.user_ns['result']
    assert result[0] == 1
    ipython_with_magic.run_cell(textwrap.dedent('\n        def rmagic_addone(u):\n            %R -i u -o result result <- u+1\n            return result[0]\n        '))
    ipython_with_magic.run_cell('result = rmagic_addone(1)')
    result = ipython_with_magic.user_ns['result']
    assert result == 2
    with pytest.raises(NameError):
        ipython_with_magic.run_line_magic('R', '-i var_not_defined 1+1')