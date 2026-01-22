from sys import version_info
from pyflakes import messages as m
from pyflakes.test.harness import TestCase, skip, skipIf
def test_withStatementUndefinedInExpression(self):
    """
        An undefined name warning is emitted if a name in the I{test}
        expression of a C{with} statement is undefined.
        """
    self.flakes('\n        with bar as baz:\n            pass\n        ', m.UndefinedName)
    self.flakes('\n        with bar as bar:\n            pass\n        ', m.UndefinedName)