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
def test_latex_completions(self):
    from IPython.core.latex_symbols import latex_symbols
    import random
    ip = get_ipython()
    keys = random.sample(sorted(latex_symbols), 10)
    for k in keys:
        text, matches = ip.complete(k)
        self.assertEqual(text, k)
        self.assertEqual(matches, [latex_symbols[k]])
    text, matches = ip.complete('print(\\alpha')
    self.assertEqual(text, '\\alpha')
    self.assertEqual(matches[0], latex_symbols['\\alpha'])
    text, matches = ip.complete('\\al')
    self.assertIn('\\alpha', matches)
    self.assertIn('\\aleph', matches)