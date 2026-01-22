import os
from pathlib import Path
import re
import sys
import tempfile
import unittest
from contextlib import contextmanager
from io import StringIO
from subprocess import Popen, PIPE
from unittest.mock import patch
from traitlets.config.loader import Config
from IPython.utils.process import get_output_error_code
from IPython.utils.text import list_strings
from IPython.utils.io import temp_pyfile, Tee
from IPython.utils import py3compat
from . import decorators as dec
from . import skipdoctest
def check_pairs(func, pairs):
    """Utility function for the common case of checking a function with a
    sequence of input/output pairs.

    Parameters
    ----------
    func : callable
      The function to be tested. Should accept a single argument.
    pairs : iterable
      A list of (input, expected_output) tuples.

    Returns
    -------
    None. Raises an AssertionError if any output does not match the expected
    value.
    """
    __tracebackhide__ = True
    name = getattr(func, 'func_name', getattr(func, '__name__', '<unknown>'))
    for inp, expected in pairs:
        out = func(inp)
        assert out == expected, pair_fail_msg.format(name, inp, expected, out)