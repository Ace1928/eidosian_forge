import ast
from pyflakes import messages as m, checker
from pyflakes.test.harness import TestCase, skip
def test_global_reset_name_only(self):
    """A global statement does not prevent other names being undefined."""
    self.flakes('\n        def f1():\n            s\n\n        def f2():\n            global m\n        ', m.UndefinedName)