import re
from .. import bzr, commands, controldir, errors, option, registry
from ..builtins import cmd_commit
from ..bzr import knitrepo
from ..commands import parse_args
from . import TestCase
def test_short_name(self):
    registry = controldir.ControlDirFormatRegistry()
    opt = option.RegistryOption('format', help='', registry=registry)
    self.assertEqual(None, opt.short_name())
    opt = option.RegistryOption('format', short_name='F', help='', registry=registry)
    self.assertEqual('F', opt.short_name())