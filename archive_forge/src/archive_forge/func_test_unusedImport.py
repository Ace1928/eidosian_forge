from pyflakes import messages as m
from pyflakes.checker import (
from pyflakes.test.harness import TestCase, skip
def test_unusedImport(self):
    self.flakes('import fu, bar', m.UnusedImport, m.UnusedImport)
    self.flakes('from baz import fu, bar', m.UnusedImport, m.UnusedImport)