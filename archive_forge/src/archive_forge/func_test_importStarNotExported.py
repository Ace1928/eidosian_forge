from pyflakes import messages as m
from pyflakes.checker import (
from pyflakes.test.harness import TestCase, skip
def test_importStarNotExported(self):
    """Report unused import when not needed to satisfy __all__."""
    self.flakes("\n        from foolib import *\n        a = 1\n        __all__ = ['a']\n        ", m.ImportStarUsed, m.UnusedImport)