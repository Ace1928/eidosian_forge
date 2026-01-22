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
def ipexec(fname, options=None, commands=()):
    """Utility to call 'ipython filename'.

    Starts IPython with a minimal and safe configuration to make startup as fast
    as possible.

    Note that this starts IPython in a subprocess!

    Parameters
    ----------
    fname : str, Path
      Name of file to be executed (should have .py or .ipy extension).

    options : optional, list
      Extra command-line flags to be passed to IPython.

    commands : optional, list
      Commands to send in on stdin

    Returns
    -------
    ``(stdout, stderr)`` of ipython subprocess.
    """
    __tracebackhide__ = True
    if options is None:
        options = []
    cmdargs = default_argv() + options
    test_dir = os.path.dirname(__file__)
    ipython_cmd = get_ipython_cmd()
    full_fname = os.path.join(test_dir, fname)
    full_cmd = ipython_cmd + cmdargs + ['--', full_fname]
    env = os.environ.copy()
    env['PYTHONWARNINGS'] = 'ignore'
    env.pop('PYCHARM_HOSTED', None)
    for k, v in env.items():
        if not isinstance(v, str):
            print(k, v)
    p = Popen(full_cmd, stdout=PIPE, stderr=PIPE, stdin=PIPE, env=env)
    out, err = p.communicate(input=py3compat.encode('\n'.join(commands)) or None)
    out, err = (py3compat.decode(out), py3compat.decode(err))
    if out:
        out = re.sub('\\x1b\\[[^h]+h', '', out)
    return (out, err)