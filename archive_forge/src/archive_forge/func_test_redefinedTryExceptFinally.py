from pyflakes import messages as m
from pyflakes.checker import (
from pyflakes.test.harness import TestCase, skip
def test_redefinedTryExceptFinally(self):
    self.flakes('\n        try:\n            from aa import a\n        except ImportError:\n            from bb import a\n        finally:\n            a = 42\n        print(a)\n        ')