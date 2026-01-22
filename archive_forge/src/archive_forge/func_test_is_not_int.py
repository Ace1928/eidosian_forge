from pyflakes.messages import IsLiteral
from pyflakes.test.harness import TestCase
def test_is_not_int(self):
    self.flakes('\n        x = 10\n        if x is not 10:\n            pass\n        ', IsLiteral)