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
def parse_test_output(txt):
    """Parse the output of a test run and return errors, failures.

    Parameters
    ----------
    txt : str
      Text output of a test run, assumed to contain a line of one of the
      following forms::

        'FAILED (errors=1)'
        'FAILED (failures=1)'
        'FAILED (errors=1, failures=1)'

    Returns
    -------
    nerr, nfail
      number of errors and failures.
    """
    err_m = re.search('^FAILED \\(errors=(\\d+)\\)', txt, re.MULTILINE)
    if err_m:
        nerr = int(err_m.group(1))
        nfail = 0
        return (nerr, nfail)
    fail_m = re.search('^FAILED \\(failures=(\\d+)\\)', txt, re.MULTILINE)
    if fail_m:
        nerr = 0
        nfail = int(fail_m.group(1))
        return (nerr, nfail)
    both_m = re.search('^FAILED \\(errors=(\\d+), failures=(\\d+)\\)', txt, re.MULTILINE)
    if both_m:
        nerr = int(both_m.group(1))
        nfail = int(both_m.group(2))
        return (nerr, nfail)
    return (0, 0)