from sys import version_info
from pyflakes import messages as m
from pyflakes.test.harness import TestCase, skip, skipIf
def test_functionRedefinedAsClass(self):
    """
        If a function is redefined as a class, a warning is emitted.
        """
    self.flakes('\n        def Foo():\n            pass\n        class Foo:\n            pass\n        ', m.RedefinedWhileUnused)