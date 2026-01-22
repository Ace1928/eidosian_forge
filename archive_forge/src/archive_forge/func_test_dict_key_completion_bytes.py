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
def test_dict_key_completion_bytes(self):
    """Test handling of bytes in dict key completion"""
    ip = get_ipython()
    complete = ip.Completer.complete
    ip.user_ns['d'] = {'abc': None, b'abd': None}
    _, matches = complete(line_buffer='d[')
    self.assertIn("'abc'", matches)
    self.assertIn("b'abd'", matches)
    if False:
        _, matches = complete(line_buffer='d[b')
        self.assertIn("b'abd'", matches)
        self.assertNotIn("b'abc'", matches)
        _, matches = complete(line_buffer="d[b'")
        self.assertIn('abd', matches)
        self.assertNotIn('abc', matches)
        _, matches = complete(line_buffer="d[B'")
        self.assertIn('abd', matches)
        self.assertNotIn('abc', matches)
        _, matches = complete(line_buffer="d['")
        self.assertIn('abc', matches)
        self.assertNotIn('abd', matches)