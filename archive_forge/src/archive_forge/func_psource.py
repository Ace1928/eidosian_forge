import gc
import re
import sys
from IPython.core import page
from IPython.core.error import StdinNotImplementedError, UsageError
from IPython.core.magic import Magics, magics_class, line_magic
from IPython.testing.skipdoctest import skip_doctest
from IPython.utils.encoding import DEFAULT_ENCODING
from IPython.utils.openpy import read_py_file
from IPython.utils.path import get_py_filename
@line_magic
def psource(self, parameter_s='', namespaces=None):
    """Print (or run through pager) the source code for an object."""
    if not parameter_s:
        raise UsageError('Missing object name.')
    self.shell._inspect('psource', parameter_s, namespaces)