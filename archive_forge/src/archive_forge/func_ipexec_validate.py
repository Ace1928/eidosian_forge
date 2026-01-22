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
def ipexec_validate(fname, expected_out, expected_err='', options=None, commands=()):
    """Utility to call 'ipython filename' and validate output/error.

    This function raises an AssertionError if the validation fails.

    Note that this starts IPython in a subprocess!

    Parameters
    ----------
    fname : str, Path
      Name of the file to be executed (should have .py or .ipy extension).

    expected_out : str
      Expected stdout of the process.

    expected_err : optional, str
      Expected stderr of the process.

    options : optional, list
      Extra command-line flags to be passed to IPython.

    Returns
    -------
    None
    """
    __tracebackhide__ = True
    out, err = ipexec(fname, options, commands)
    if err:
        if expected_err:
            assert '\n'.join(err.strip().splitlines()) == '\n'.join(expected_err.strip().splitlines())
        else:
            raise ValueError('Running file %r produced error: %r' % (fname, err))
    assert '\n'.join(out.strip().splitlines()) == '\n'.join(expected_out.strip().splitlines())