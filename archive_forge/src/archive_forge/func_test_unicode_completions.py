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
def test_unicode_completions(self):
    ip = get_ipython()
    s = ['ru', '%ru', 'cd /', 'floa', 'float(x)/']
    for t in s + list(map(str, s)):
        text, matches = ip.complete(t)
        self.assertIsInstance(text, str)
        self.assertIsInstance(matches, list)