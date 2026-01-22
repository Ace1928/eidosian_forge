from sys import version_info
from pyflakes import messages as m
from pyflakes.test.harness import TestCase, skip, skipIf
def test_classWithYield(self):
    """
        If a yield is used inside a class, a warning is emitted.
        """
    self.flakes('\n        class Foo(object):\n            yield\n        ', m.YieldOutsideFunction)