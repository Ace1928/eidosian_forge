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
@pytest.mark.xfail(sys.version_info.releaselevel in ('alpha',), reason='Parso does not yet parse 3.13')
def test_all_completions_dups(self):
    """
        Make sure the output of `IPCompleter.all_completions` does not have
        duplicated prefixes.
        """
    ip = get_ipython()
    c = ip.Completer
    ip.ex('class TestClass():\n\ta=1\n\ta1=2')
    for jedi_status in [True, False]:
        with provisionalcompleter():
            ip.Completer.use_jedi = jedi_status
            matches = c.all_completions('TestCl')
            assert matches == ['TestClass'], (jedi_status, matches)
            matches = c.all_completions('TestClass.')
            assert len(matches) > 2, (jedi_status, matches)
            matches = c.all_completions('TestClass.a')
            assert matches == ['TestClass.a', 'TestClass.a1'], jedi_status