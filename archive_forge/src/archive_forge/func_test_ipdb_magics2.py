import builtins
import os
import sys
import platform
from tempfile import NamedTemporaryFile
from textwrap import dedent
from unittest.mock import patch
from IPython.core import debugger
from IPython.testing import IPYTHON_TESTING_TIMEOUT_SCALE
from IPython.testing.decorators import skip_win32
import pytest
def test_ipdb_magics2():
    """Test ipdb with a very short function.
    
    >>> old_trace = sys.gettrace()

    >>> def bar():
    ...     pass

    Run ipdb.

    >>> with PdbTestInput([
    ...    'continue',
    ... ]):
    ...     debugger.Pdb().runcall(bar)
    > <doctest ...>(2)bar()
          1 def bar():
    ----> 2    pass
    <BLANKLINE>
    ipdb> continue
    
    Restore previous trace function, e.g. for coverage.py    
    
    >>> sys.settrace(old_trace)
    """