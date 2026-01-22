import errno
import inspect
import sys
from .. import builtins, commands, config, errors, option, tests, trace
from ..commands import display_command
from . import TestSkipped
def test_pre_command_error(self):
    """Ensure an CommandError in pre_command aborts the command"""
    hook_calls = []

    def pre_command(cmd):
        hook_calls.append('pre')
        raise errors.CommandError()

    def post_command(cmd, e):
        self.fail('post_command should not be called')

    def run(cmd):
        self.fail('command should not be called')
    self.overrideAttr(builtins.cmd_rocks, 'run', run)
    commands.install_bzr_command_hooks()
    commands.Command.hooks.install_named_hook('pre_command', pre_command, None)
    commands.Command.hooks.install_named_hook('post_command', post_command, None)
    self.assertEqual([], hook_calls)
    self.assertRaises(errors.CommandError, commands.run_bzr, ['rocks'])
    self.assertEqual(['pre'], hook_calls)