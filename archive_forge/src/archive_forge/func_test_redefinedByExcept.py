from pyflakes import messages as m
from pyflakes.checker import (
from pyflakes.test.harness import TestCase, skip
def test_redefinedByExcept(self):
    expected = [m.RedefinedWhileUnused]
    expected.append(m.UnusedVariable)
    self.flakes('\n        import fu\n        try: pass\n        except Exception as fu: pass\n        ', *expected)