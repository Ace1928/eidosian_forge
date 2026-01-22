from pyflakes.messages import IsLiteral
from pyflakes.test.harness import TestCase
def test_left_is_str(self):
    self.flakes("\n        x = 'foo'\n        if 'foo' is x:\n            pass\n        ", IsLiteral)