from sys import version_info
from pyflakes import messages as m
from pyflakes.test.harness import TestCase, skip, skipIf
def test_withStatementSingleName(self):
    """
        No warnings are emitted for using a name defined by a C{with} statement
        within the suite or afterwards.
        """
    self.flakes("\n        with open('foo') as bar:\n            bar\n        bar\n        ")