import ast
from pyflakes import messages as m, checker
from pyflakes.test.harness import TestCase, skip
def test_magicGlobalsPath(self):
    """
        Use of the C{__path__} magic global should not emit an undefined name
        warning, if you refer to it from a file called __init__.py.
        """
    self.flakes('__path__', m.UndefinedName)
    self.flakes('__path__', filename='package/__init__.py')