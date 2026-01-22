from sys import version_info
from pyflakes import messages as m
from pyflakes.test.harness import TestCase, skip, skipIf
def test_attrAugmentedAssignment(self):
    """
        Augmented assignment of attributes is supported.
        We don't care about attr refs.
        """
    self.flakes('\n        foo = None\n        foo.bar += foo.baz\n        ')