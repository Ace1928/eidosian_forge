from pyflakes.messages import IsLiteral
from pyflakes.test.harness import TestCase
def test_left_is_not_false(self):
    self.flakes('\n        x = False\n        if False is not x:\n            pass\n        ')