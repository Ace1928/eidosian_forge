from pyflakes.messages import IsLiteral
from pyflakes.test.harness import TestCase
def test_is_not_false(self):
    self.flakes('\n        x = False\n        if x is not False:\n            pass\n        ')