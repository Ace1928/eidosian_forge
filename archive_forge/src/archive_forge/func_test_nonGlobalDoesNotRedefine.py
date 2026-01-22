from pyflakes import messages as m
from pyflakes.checker import (
from pyflakes.test.harness import TestCase, skip
def test_nonGlobalDoesNotRedefine(self):
    self.flakes('\n        import fu\n        def a():\n            fu = 3\n            return fu\n        fu\n        ')