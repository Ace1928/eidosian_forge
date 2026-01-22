from pyflakes import messages as m
from pyflakes.checker import (
from pyflakes.test.harness import TestCase, skip
def test_tryingMultipleImports(self):
    self.flakes('\n        try:\n            import fu\n        except ImportError:\n            import bar as fu\n        fu\n        ')