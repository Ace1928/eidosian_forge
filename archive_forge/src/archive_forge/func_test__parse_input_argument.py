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
@pytest.mark.parametrize('arg,expected', (('foo', ('foo', 'foo')), ('bar=foo', ('bar', 'foo')), ('bar=baz.foo', ('bar', 'baz.foo'))))
def test__parse_input_argument(arg, expected):
    assert expected == rmagic._parse_input_argument(arg)