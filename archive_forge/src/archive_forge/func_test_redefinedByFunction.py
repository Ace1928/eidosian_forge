from pyflakes import messages as m
from pyflakes.checker import (
from pyflakes.test.harness import TestCase, skip
def test_redefinedByFunction(self):
    self.flakes('\n        import fu\n        def fu():\n            pass\n        ', m.RedefinedWhileUnused)