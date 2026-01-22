import os
import pytest
import sys
import textwrap
import unittest
from contextlib import contextmanager
from traitlets.config.loader import Config
from IPython import get_ipython
from IPython.core import completer
from IPython.utils.tempdir import TemporaryDirectory, TemporaryWorkingDirectory
from IPython.utils.generics import complete_object
from IPython.testing import decorators as dec
from IPython.core.completer import (
@pytest.mark.xfail(sys.platform == 'win32', reason='abspath completions fail on Windows')
def test_abspath_file_completions(self):
    ip = get_ipython()
    with TemporaryDirectory() as tmpdir:
        prefix = os.path.join(tmpdir, 'foo')
        suffixes = ['1', '2']
        names = [prefix + s for s in suffixes]
        for n in names:
            open(n, 'w', encoding='utf-8').close()
        c = ip.complete(prefix)[1]
        self.assertEqual(c, names)
        cmd = 'a = f("%s' % prefix
        c = ip.complete(prefix, cmd)[1]
        comp = [prefix + s for s in suffixes]
        self.assertEqual(c, comp)