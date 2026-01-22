import errno
import inspect
import sys
from .. import builtins, commands, config, errors, option, tests, trace
from ..commands import display_command
from . import TestSkipped
def hook_missing(self):
    """Hook get_missing_command for testing."""
    self.hook_calls = []

    class ACommand(commands.Command):
        __doc__ = 'A sample command.'

    def get_missing_cmd(cmd_name):
        self.hook_calls.append(('called', cmd_name))
        if cmd_name in ('foo', 'info'):
            return ACommand()
    commands.Command.hooks.install_named_hook('get_missing_command', get_missing_cmd, None)
    self.ACommand = ACommand