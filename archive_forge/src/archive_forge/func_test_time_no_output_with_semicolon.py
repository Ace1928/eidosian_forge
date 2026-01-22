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
def test_time_no_output_with_semicolon():
    ip = get_ipython()
    with tt.AssertPrints(' 123456'):
        with tt.AssertPrints('Wall time: ', suppress=False):
            with tt.AssertPrints('CPU times: ', suppress=False):
                ip.run_cell('%time 123000+456')
    with tt.AssertNotPrints(' 123456'):
        with tt.AssertPrints('Wall time: ', suppress=False):
            with tt.AssertPrints('CPU times: ', suppress=False):
                ip.run_cell('%time 123000+456;')
    with tt.AssertPrints(' 123456'):
        with tt.AssertPrints('Wall time: ', suppress=False):
            with tt.AssertPrints('CPU times: ', suppress=False):
                ip.run_cell('%time 123000+456 # Comment')
    with tt.AssertNotPrints(' 123456'):
        with tt.AssertPrints('Wall time: ', suppress=False):
            with tt.AssertPrints('CPU times: ', suppress=False):
                ip.run_cell('%time 123000+456; # Comment')
    with tt.AssertPrints(' 123456'):
        with tt.AssertPrints('Wall time: ', suppress=False):
            with tt.AssertPrints('CPU times: ', suppress=False):
                ip.run_cell('%time 123000+456 # ;Comment')
    with tt.AssertPrints('123456'):
        with tt.AssertPrints('Wall time: ', suppress=False):
            with tt.AssertPrints('CPU times: ', suppress=False):
                ip.run_cell('%%time\n123000+456\n\n\n')
    with tt.AssertNotPrints('123456'):
        with tt.AssertPrints('Wall time: ', suppress=False):
            with tt.AssertPrints('CPU times: ', suppress=False):
                ip.run_cell('%%time\n123000+456;\n\n\n')
    with tt.AssertPrints('123456'):
        with tt.AssertPrints('Wall time: ', suppress=False):
            with tt.AssertPrints('CPU times: ', suppress=False):
                ip.run_cell('%%time\n123000+456  # Comment\n\n\n')
    with tt.AssertNotPrints('123456'):
        with tt.AssertPrints('Wall time: ', suppress=False):
            with tt.AssertPrints('CPU times: ', suppress=False):
                ip.run_cell('%%time\n123000+456;  # Comment\n\n\n')
    with tt.AssertPrints('123456'):
        with tt.AssertPrints('Wall time: ', suppress=False):
            with tt.AssertPrints('CPU times: ', suppress=False):
                ip.run_cell('%%time\n123000+456  # ;Comment\n\n\n')