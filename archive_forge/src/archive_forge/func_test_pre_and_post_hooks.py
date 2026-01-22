import errno
import inspect
import sys
from .. import builtins, commands, config, errors, option, tests, trace
from ..commands import display_command
from . import TestSkipped
def test_pre_and_post_hooks(self):
    hook_calls = []

    def pre_command(cmd):
        self.assertEqual([], hook_calls)
        hook_calls.append('pre')

    def post_command(cmd):
        self.assertEqual(['pre', 'run'], hook_calls)
        hook_calls.append('post')

    def run(cmd):
        self.assertEqual(['pre'], hook_calls)
        hook_calls.append('run')
    self.overrideAttr(builtins.cmd_rocks, 'run', run)
    commands.install_bzr_command_hooks()
    commands.Command.hooks.install_named_hook('pre_command', pre_command, None)
    commands.Command.hooks.install_named_hook('post_command', post_command, None)
    self.assertEqual([], hook_calls)
    self.run_bzr(['rocks', '-Oxx=12', '-Oyy=foo'])
    self.assertEqual(['pre', 'run', 'post'], hook_calls)