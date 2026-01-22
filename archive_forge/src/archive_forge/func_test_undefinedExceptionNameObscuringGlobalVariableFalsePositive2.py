import ast
from pyflakes import messages as m, checker
from pyflakes.test.harness import TestCase, skip
def test_undefinedExceptionNameObscuringGlobalVariableFalsePositive2(self):
    """Exception names obscure globals, can't be used after. Unless.

        Last line will never raise NameError because `error` is only
        falsy if the `except:` block has not been entered."""
    self.flakes("\n        exc = 'Original value'\n        def func():\n            global exc\n            error = None\n            try:\n                raise ValueError('ve')\n            except ValueError as exc:\n                error = 'exception logged'\n            if error:\n                print(error)\n            else:\n                exc\n        ", m.UnusedVariable)