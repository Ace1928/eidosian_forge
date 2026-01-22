import re
from .. import bzr, commands, controldir, errors, option, registry
from ..builtins import cmd_commit
from ..bzr import knitrepo
from ..commands import parse_args
from . import TestCase
def test_lazy_registry(self):
    options = [option.RegistryOption('format', '', lazy_registry=('breezy.controldir', 'format_registry'), converter=str)]
    opts, args = self.parse(options, ['--format', 'knit'])
    self.assertEqual({'format': 'knit'}, opts)
    self.assertRaises(option.BadOptionValue, self.parse, options, ['--format', 'BAD'])