import re
from .. import bzr, commands, controldir, errors, option, registry
from ..builtins import cmd_commit
from ..bzr import knitrepo
from ..commands import parse_args
from . import TestCase
def test_option_grammar(self):
    msgs = []
    option_re = re.compile('^[A-Z][^\\n]+\\.(?: \\([^\\n]+\\))?$')
    for scope, opt in self.get_builtin_command_options():
        for name, _, _, helptxt in opt.iter_switches():
            if name != opt.name:
                name = '/'.join([opt.name, name])
            if not helptxt:
                msgs.append('%-16s %-16s %s' % (scope or 'GLOBAL', name, 'NO HELP'))
            elif not option_re.match(helptxt):
                msgs.append('%-16s %-16s %s' % (scope or 'GLOBAL', name, helptxt))
    if msgs:
        self.fail("The following options don't match the style guide:\n" + '\n'.join(msgs))