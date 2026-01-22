from sys import version_info
from pyflakes import messages as m
from pyflakes.test.harness import TestCase, skip, skipIf
def test_classNameDefinedPreviously(self):
    """
        If a class name is used in the body of that class's definition and
        the name was previously defined in some other way, no warning is
        emitted.
        """
    self.flakes('\n        foo = None\n        class foo:\n            foo\n        ')