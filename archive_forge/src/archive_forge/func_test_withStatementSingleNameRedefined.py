from sys import version_info
from pyflakes import messages as m
from pyflakes.test.harness import TestCase, skip, skipIf
def test_withStatementSingleNameRedefined(self):
    """
        A redefined name warning is emitted if a name bound by an import is
        rebound by the name defined by a C{with} statement.
        """
    self.flakes("\n        import bar\n        with open('foo') as bar:\n            pass\n        ", m.RedefinedWhileUnused)