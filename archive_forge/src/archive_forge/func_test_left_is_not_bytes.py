from pyflakes.messages import IsLiteral
from pyflakes.test.harness import TestCase
def test_left_is_not_bytes(self):
    self.flakes("\n        x = b'foo'\n        if b'foo' is not x:\n            pass\n        ", IsLiteral)