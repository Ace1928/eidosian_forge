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
def can_quit():
    """Test that quit work in ipydb

    >>> old_trace = sys.gettrace()

    >>> def bar():
    ...     pass

    >>> with PdbTestInput([
    ...    'quit',
    ... ]):
    ...     debugger.Pdb().runcall(bar)
    > <doctest ...>(2)bar()
            1 def bar():
    ----> 2    pass
    <BLANKLINE>
    ipdb> quit

    Restore previous trace function, e.g. for coverage.py

    >>> sys.settrace(old_trace)
    """