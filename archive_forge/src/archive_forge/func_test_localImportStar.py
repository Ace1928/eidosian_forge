from pyflakes import messages as m
from pyflakes.checker import (
from pyflakes.test.harness import TestCase, skip
def test_localImportStar(self):
    """import * is only allowed at module level."""
    self.flakes('\n        def a():\n            from fu import *\n        ', m.ImportStarNotPermitted)
    self.flakes('\n        class a:\n            from fu import *\n        ', m.ImportStarNotPermitted)
    checker = self.flakes('\n        class a:\n            from .. import *\n        ', m.ImportStarNotPermitted)
    error = checker.messages[0]
    assert error.message == "'from %s import *' only allowed at module level"
    assert error.message_args == ('..',)