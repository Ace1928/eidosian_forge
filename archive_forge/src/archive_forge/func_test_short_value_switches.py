import re
from .. import bzr, commands, controldir, errors, option, registry
from ..builtins import cmd_commit
from ..bzr import knitrepo
from ..commands import parse_args
from . import TestCase
def test_short_value_switches(self):
    reg = registry.Registry()
    reg.register('short', 'ShortChoice')
    reg.register('long', 'LongChoice')
    ropt = option.RegistryOption('choice', '', reg, value_switches=True, short_value_switches={'short': 's'})
    opts, args = parse([ropt], ['--short'])
    self.assertEqual('ShortChoice', opts.choice)
    opts, args = parse([ropt], ['-s'])
    self.assertEqual('ShortChoice', opts.choice)