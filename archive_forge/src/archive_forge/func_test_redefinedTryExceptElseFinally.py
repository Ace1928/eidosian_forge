from pyflakes import messages as m
from pyflakes.checker import (
from pyflakes.test.harness import TestCase, skip
def test_redefinedTryExceptElseFinally(self):
    self.flakes('\n        try:\n            import b\n        except ImportError:\n            b = Ellipsis\n            from bb import a\n        else:\n            from aa import a\n        finally:\n            a = 42\n        print(a, b)\n        ')