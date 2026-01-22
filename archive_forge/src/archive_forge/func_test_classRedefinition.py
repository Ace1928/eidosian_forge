from sys import version_info
from pyflakes import messages as m
from pyflakes.test.harness import TestCase, skip, skipIf
def test_classRedefinition(self):
    """
        If a class is defined twice in the same module, a warning is emitted.
        """
    self.flakes('\n        class Foo:\n            pass\n        class Foo:\n            pass\n        ', m.RedefinedWhileUnused)