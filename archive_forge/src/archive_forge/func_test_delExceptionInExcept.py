import ast
from pyflakes import messages as m, checker
from pyflakes.test.harness import TestCase, skip
def test_delExceptionInExcept(self):
    """The exception name can be deleted in the except: block."""
    self.flakes('\n        try:\n            pass\n        except Exception as exc:\n            del exc\n        ')