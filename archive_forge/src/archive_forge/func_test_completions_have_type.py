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
def test_completions_have_type(self):
    """
        Lets make sure matchers provide completion type.
        """
    ip = get_ipython()
    with provisionalcompleter():
        ip.Completer.use_jedi = False
        completions = ip.Completer.completions('%tim', 3)
        c = next(completions)
    assert c.type == 'magic', 'Type of magic was not assigned by completer'