from pyflakes import messages as m
from pyflakes.checker import (
from pyflakes.test.harness import TestCase, skip
def test_unusedImport_relative(self):
    self.flakes('from . import fu', m.UnusedImport)
    self.flakes('from . import fu as baz', m.UnusedImport)
    self.flakes('from .. import fu', m.UnusedImport)
    self.flakes('from ... import fu', m.UnusedImport)
    self.flakes('from .. import fu as baz', m.UnusedImport)
    self.flakes('from .bar import fu', m.UnusedImport)
    self.flakes('from ..bar import fu', m.UnusedImport)
    self.flakes('from ...bar import fu', m.UnusedImport)
    self.flakes('from ...bar import fu as baz', m.UnusedImport)
    checker = self.flakes('from . import fu', m.UnusedImport)
    error = checker.messages[0]
    assert error.message == '%r imported but unused'
    assert error.message_args == ('.fu',)
    checker = self.flakes('from . import fu as baz', m.UnusedImport)
    error = checker.messages[0]
    assert error.message == '%r imported but unused'
    assert error.message_args == ('.fu as baz',)