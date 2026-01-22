from sys import version_info
from pyflakes import messages as m
from pyflakes.test.harness import TestCase, skip, skipIf
def test_assert_tuple(self):
    """An assert of a non-empty tuple is always True."""
    self.flakes("\n        assert (False, 'x')\n        assert (False, )\n        ", m.AssertTuple, m.AssertTuple)