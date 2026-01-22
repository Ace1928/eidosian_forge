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
def test_fwd_unicode_restricts(self):
    ip = get_ipython()
    completer = ip.Completer
    text = '\\ROMAN NUMERAL FIVE'
    with provisionalcompleter():
        completer.use_jedi = True
        completions = [completion.text for completion in completer.completions(text, len(text))]
        self.assertEqual(completions, ['Ⅴ'])