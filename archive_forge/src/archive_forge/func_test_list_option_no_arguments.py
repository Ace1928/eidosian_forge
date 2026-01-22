import re
from .. import bzr, commands, controldir, errors, option, registry
from ..builtins import cmd_commit
from ..bzr import knitrepo
from ..commands import parse_args
from . import TestCase
def test_list_option_no_arguments(self):
    options = [option.ListOption('hello', type=str)]
    opts, args = self.parse(options, [])
    self.assertEqual([], opts.hello)