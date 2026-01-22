from sys import version_info
from pyflakes import messages as m
from pyflakes.test.harness import TestCase, skip, skipIf
def test_doubleClosedOver(self):
    """
        Don't warn when the assignment is used in an inner function, even if
        that inner function itself is in an inner function.
        """
    self.flakes('\n        def barMaker():\n            foo = 5\n            def bar():\n                def baz():\n                    return foo\n            return bar\n        ')