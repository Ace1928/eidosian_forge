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
def help_output_test(subcommand=''):
    """test that `ipython [subcommand] -h` works"""
    cmd = get_ipython_cmd() + [subcommand, '-h']
    out, err, rc = get_output_error_code(cmd)
    assert rc == 0, err
    assert 'Traceback' not in err
    assert 'Options' in out
    assert '--help-all' in out
    return (out, err)