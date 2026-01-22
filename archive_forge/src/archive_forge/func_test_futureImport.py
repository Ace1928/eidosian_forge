from pyflakes import messages as m
from pyflakes.checker import (
from pyflakes.test.harness import TestCase, skip
def test_futureImport(self):
    """__future__ is special."""
    self.flakes('from __future__ import division')
    self.flakes('\n        "docstring is allowed before future import"\n        from __future__ import division\n        ')