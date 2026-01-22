import ast
from pyflakes import messages as m, checker
from pyflakes.test.harness import TestCase, skip
@skip('error reporting disabled due to false positives below')
def test_undefinedExceptionNameObscuringGlobalVariable2(self):
    """Exception names obscure globals, can't be used after.

        Last line will raise NameError on Python 3 because the name is
        locally unbound after the `except:` block, even if it's
        nonlocal. We should issue an error in this case because code
        only working correctly if an exception isn't raised, is invalid.
        Unless it's explicitly silenced, see false positives below."""
    self.flakes("\n        exc = 'Original value'\n        def func():\n            global exc\n            try:\n                raise ValueError('ve')\n            except ValueError as exc:\n                pass  # block never entered, exc stays unbound\n            exc\n        ", m.UndefinedLocal)