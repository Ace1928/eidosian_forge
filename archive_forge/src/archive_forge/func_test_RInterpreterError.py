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
def test_RInterpreterError():
    line = 123
    err = 'Arrh!'
    stdout = 'Kaput'
    rie = rmagic.RInterpreterError(line, err, stdout)
    assert str(rie).startswith(rie.msg_prefix_template % (line, err))