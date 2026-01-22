from sys import version_info
from pyflakes import messages as m
from pyflakes.test.harness import TestCase, skip, skipIf
def test_ifexp(self):
    """
        Test C{foo if bar else baz} statements.
        """
    self.flakes("a = 'moo' if True else 'oink'")
    self.flakes("a = foo if True else 'oink'", m.UndefinedName)
    self.flakes("a = 'moo' if True else bar", m.UndefinedName)