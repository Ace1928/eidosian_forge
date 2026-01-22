import ast
from pyflakes import messages as m, checker
from pyflakes.test.harness import TestCase, skip
@skip('error reporting disabled due to false positives below')
def test_undefinedExceptionNameObscuringLocalVariable(self):
    """Exception names obscure locals, can't be used after.

        Last line will raise UnboundLocalError on Python 3 after exiting
        the except: block. Note next two examples for false positives to
        watch out for."""
    self.flakes("\n        exc = 'Original value'\n        try:\n            raise ValueError('ve')\n        except ValueError as exc:\n            pass\n        exc\n        ", m.UndefinedName)