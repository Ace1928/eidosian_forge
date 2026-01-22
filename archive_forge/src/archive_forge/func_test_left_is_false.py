from pyflakes.messages import IsLiteral
from pyflakes.test.harness import TestCase
def test_left_is_false(self):
    self.flakes('\n        x = False\n        if False is x:\n            pass\n        ')