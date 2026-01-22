from pyflakes.messages import IsLiteral
from pyflakes.test.harness import TestCase
def test_chained_operators_is_true_end(self):
    self.flakes('\n        x = 5\n        if 4 < x is True:\n            pass\n        ')