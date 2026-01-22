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
def test_magic_color(self):
    ip = get_ipython()
    c = ip.Completer
    s, matches = c.complete(None, 'colo')
    self.assertIn('%colors', matches)
    s, matches = c.complete(None, 'colo')
    self.assertNotIn('NoColor', matches)
    s, matches = c.complete(None, '%colors')
    self.assertNotIn('NoColor', matches)
    s, matches = c.complete(None, 'colors ')
    self.assertIn('NoColor', matches)
    s, matches = c.complete(None, '%colors ')
    self.assertIn('NoColor', matches)
    s, matches = c.complete(None, 'colors NoCo')
    self.assertListEqual(['NoColor'], matches)
    s, matches = c.complete(None, '%colors NoCo')
    self.assertListEqual(['NoColor'], matches)