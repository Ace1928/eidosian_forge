from pyflakes.messages import IsLiteral
from pyflakes.test.harness import TestCase
def test_left_is_true(self):
    self.flakes('\n        x = True\n        if True is x:\n            pass\n        ')