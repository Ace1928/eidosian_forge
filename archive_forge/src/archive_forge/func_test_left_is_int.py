from pyflakes.messages import IsLiteral
from pyflakes.test.harness import TestCase
def test_left_is_int(self):
    self.flakes('\n        x = 10\n        if 10 is x:\n            pass\n        ', IsLiteral)