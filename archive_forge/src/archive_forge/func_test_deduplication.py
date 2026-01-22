import errno
import inspect
import sys
from .. import builtins, commands, config, errors, option, tests, trace
from ..commands import display_command
from . import TestSkipped
def test_deduplication(self):
    """Duplicates in _see_also are stripped out."""
    command = self._get_command_with_see_also(['foo', 'foo'])
    self.assertEqual(['foo'], command.get_see_also())