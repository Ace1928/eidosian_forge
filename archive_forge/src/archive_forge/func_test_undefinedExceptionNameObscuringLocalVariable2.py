import ast
from pyflakes import messages as m, checker
from pyflakes.test.harness import TestCase, skip
def test_undefinedExceptionNameObscuringLocalVariable2(self):
    """Exception names are unbound after the `except:` block.

        Last line will raise UnboundLocalError.
        The exc variable is unused inside the exception handler.
        """
    self.flakes("\n        try:\n            raise ValueError('ve')\n        except ValueError as exc:\n            pass\n        print(exc)\n        exc = 'Original value'\n        ", m.UndefinedName, m.UnusedVariable)