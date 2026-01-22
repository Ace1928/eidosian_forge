from pyflakes import messages as m
from pyflakes.checker import (
from pyflakes.test.harness import TestCase, skip
def test_redefinedByClass(self):
    self.flakes('\n        import fu\n        class fu:\n            pass\n        ', m.RedefinedWhileUnused)