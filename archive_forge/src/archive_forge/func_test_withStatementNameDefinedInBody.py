from sys import version_info
from pyflakes import messages as m
from pyflakes.test.harness import TestCase, skip, skipIf
def test_withStatementNameDefinedInBody(self):
    """
        A name defined in the body of a C{with} statement can be used after
        the body ends without warning.
        """
    self.flakes("\n        with open('foo') as bar:\n            baz = 10\n        baz\n        ")