import re
from .. import bzr, commands, controldir, errors, option, registry
from ..builtins import cmd_commit
from ..bzr import knitrepo
from ..commands import parse_args
from . import TestCase
def test_is_hidden(self):
    registry = controldir.ControlDirFormatRegistry()
    bzr.register_metadir(registry, 'hidden', 'HiddenFormat', 'hidden help text', hidden=True)
    bzr.register_metadir(registry, 'visible', 'VisibleFormat', 'visible help text', hidden=False)
    format = option.RegistryOption('format', '', registry, str)
    self.assertTrue(format.is_hidden('hidden'))
    self.assertFalse(format.is_hidden('visible'))