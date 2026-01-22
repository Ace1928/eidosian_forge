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
def test_alias_magic():
    """Test %alias_magic."""
    ip = get_ipython()
    mm = ip.magics_manager
    ip.run_line_magic('alias_magic', 'timeit_alias timeit')
    assert 'timeit_alias' in mm.magics['line']
    assert 'timeit_alias' in mm.magics['cell']
    ip.run_line_magic('alias_magic', '--cell timeit_cell_alias timeit')
    assert 'timeit_cell_alias' not in mm.magics['line']
    assert 'timeit_cell_alias' in mm.magics['cell']
    ip.run_line_magic('alias_magic', '--line env_alias env')
    assert ip.run_line_magic('env', '') == ip.run_line_magic('env_alias', '')
    ip.run_line_magic('alias_magic', '--line history_alias history --params ' + shlex.quote('3'))
    assert 'history_alias' in mm.magics['line']