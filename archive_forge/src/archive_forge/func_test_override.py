import re
from .. import bzr, commands, controldir, errors, option, registry
from ..builtins import cmd_commit
from ..bzr import knitrepo
from ..commands import parse_args
from . import TestCase
def test_override(self):
    options = [option.Option('hello', type=str), option.Option('hi', type=str, param_name='hello')]
    opts, args = self.parse(options, ['--hello', 'a', '--hello', 'b'])
    self.assertEqual('b', opts.hello)
    opts, args = self.parse(options, ['--hello', 'b', '--hello', 'a'])
    self.assertEqual('a', opts.hello)
    opts, args = self.parse(options, ['--hello', 'a', '--hi', 'b'])
    self.assertEqual('b', opts.hello)
    opts, args = self.parse(options, ['--hi', 'b', '--hello', 'a'])
    self.assertEqual('a', opts.hello)