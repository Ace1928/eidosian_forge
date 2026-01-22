from sys import version_info
from pyflakes import messages as m
from pyflakes.test.harness import TestCase, skip, skipIf
def test_augmentedAssignmentImportedFunctionCall(self):
    """
        Consider a function that is called on the right part of an
        augassign operation to be used.
        """
    self.flakes('\n        from foo import bar\n        baz = 0\n        baz += bar()\n        ')