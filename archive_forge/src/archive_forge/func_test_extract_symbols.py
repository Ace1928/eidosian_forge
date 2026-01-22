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
def test_extract_symbols():
    source = 'import foo\na = 10\ndef b():\n    return 42\n\n\nclass A: pass\n\n\n'
    symbols_args = ['a', 'b', 'A', 'A,b', 'A,a', 'z']
    expected = [([], ['a']), (['def b():\n    return 42\n'], []), (['class A: pass\n'], []), (['class A: pass\n', 'def b():\n    return 42\n'], []), (['class A: pass\n'], ['a']), ([], ['z'])]
    for symbols, exp in zip(symbols_args, expected):
        assert code.extract_symbols(source, symbols) == exp