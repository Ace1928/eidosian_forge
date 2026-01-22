import errno
import inspect
import sys
from .. import builtins, commands, config, errors, option, tests, trace
from ..commands import display_command
from . import TestSkipped
def test_post_hook_provided_exception(self):
    hook_calls = []

    def post_command(cmd):
        hook_calls.append('post')

    def run(cmd):
        hook_calls.append('run')
        raise self.TestError()
    self.overrideAttr(builtins.cmd_rocks, 'run', run)
    commands.install_bzr_command_hooks()
    commands.Command.hooks.install_named_hook('post_command', post_command, None)
    self.assertEqual([], hook_calls)
    self.assertRaises(self.TestError, commands.run_bzr, ['rocks'])
    self.assertEqual(['run', 'post'], hook_calls)