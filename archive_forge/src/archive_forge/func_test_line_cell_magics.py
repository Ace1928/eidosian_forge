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
def test_line_cell_magics(self):
    from IPython.core.magic import register_line_cell_magic

    @register_line_cell_magic
    def _bar_cellm(line, cell):
        pass
    ip = get_ipython()
    c = ip.Completer
    s, matches = c.complete(None, '_bar_ce')
    self.assertIn('%_bar_cellm', matches)
    self.assertIn('%%_bar_cellm', matches)
    s, matches = c.complete(None, '%_bar_ce')
    self.assertIn('%_bar_cellm', matches)
    self.assertIn('%%_bar_cellm', matches)
    s, matches = c.complete(None, '%%_bar_ce')
    self.assertNotIn('%_bar_cellm', matches)
    self.assertIn('%%_bar_cellm', matches)