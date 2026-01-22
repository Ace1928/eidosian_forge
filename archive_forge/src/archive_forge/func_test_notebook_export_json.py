import gc
import io
import os
import re
import shlex
import sys
import warnings
from importlib import invalidate_caches
from io import StringIO
from pathlib import Path
from textwrap import dedent
from unittest import TestCase, mock
import pytest
from IPython import get_ipython
from IPython.core import magic
from IPython.core.error import UsageError
from IPython.core.magic import (
from IPython.core.magics import code, execution, logging, osm, script
from IPython.testing import decorators as dec
from IPython.testing import tools as tt
from IPython.utils.io import capture_output
from IPython.utils.process import find_cmd
from IPython.utils.tempdir import TemporaryDirectory, TemporaryWorkingDirectory
from IPython.utils.syspathcontext import prepended_to_syspath
from .test_debugger import PdbTestInput
from tempfile import NamedTemporaryFile
from IPython.core.magic import (
def test_notebook_export_json():
    pytest.importorskip('nbformat')
    _ip = get_ipython()
    _ip.history_manager.reset()
    cmds = ['a=1', 'def b():\n  return a**2', "print('noël, été', b())"]
    for i, cmd in enumerate(cmds, start=1):
        _ip.history_manager.store_inputs(i, cmd)
    with TemporaryDirectory() as td:
        outfile = os.path.join(td, 'nb.ipynb')
        _ip.run_line_magic('notebook', '%s' % outfile)