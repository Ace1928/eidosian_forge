import re
from .. import bzr, commands, controldir, errors, option, registry
from ..builtins import cmd_commit
from ..bzr import knitrepo
from ..commands import parse_args
from . import TestCase
def get_builtin_command_options(self):
    g = []
    commands.install_bzr_command_hooks()
    for cmd_name in sorted(commands.builtin_command_names()):
        cmd = commands.get_cmd_object(cmd_name)
        for opt_name, opt in sorted(cmd.options().items()):
            g.append((cmd_name, opt))
    self.assertTrue(g)
    return g