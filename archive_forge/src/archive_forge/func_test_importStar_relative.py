from pyflakes import messages as m
from pyflakes.checker import (
from pyflakes.test.harness import TestCase, skip
def test_importStar_relative(self):
    """Use of import * from a relative import is reported."""
    self.flakes('from .fu import *', m.ImportStarUsed, m.UnusedImport)
    self.flakes('\n        try:\n            from .fu import *\n        except:\n            pass\n        ', m.ImportStarUsed, m.UnusedImport)
    checker = self.flakes('from .fu import *', m.ImportStarUsed, m.UnusedImport)
    error = checker.messages[0]
    assert error.message.startswith("'from %s import *' used; unable ")
    assert error.message_args == ('.fu',)
    error = checker.messages[1]
    assert error.message == '%r imported but unused'
    assert error.message_args == ('.fu.*',)
    checker = self.flakes('from .. import *', m.ImportStarUsed, m.UnusedImport)
    error = checker.messages[0]
    assert error.message.startswith("'from %s import *' used; unable ")
    assert error.message_args == ('..',)
    error = checker.messages[1]
    assert error.message == '%r imported but unused'
    assert error.message_args == ('from .. import *',)