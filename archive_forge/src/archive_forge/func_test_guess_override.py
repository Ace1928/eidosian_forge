import errno
import inspect
import sys
from .. import builtins, commands, config, errors, option, tests, trace
from ..commands import display_command
from . import TestSkipped
def test_guess_override(self):
    self.assertEqual('ci', commands.guess_command('ic'))