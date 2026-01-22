import re
from .. import bzr, commands, controldir, errors, option, registry
from ..builtins import cmd_commit
from ..bzr import knitrepo
from ..commands import parse_args
from . import TestCase
def test_option_callback_list(self):
    """Test callbacks work for list options."""
    cb_calls = []

    def cb(option, name, value, parser):
        cb_calls.append((option, name, value[:], parser))
    options = [option.ListOption('hello', type=str, custom_callback=cb)]
    opts, args = self.parse(options, ['--hello=world', '--hello=mars', '--hello=-'])
    self.assertEqual(3, len(cb_calls))
    opt, name, value, parser = cb_calls[0]
    self.assertEqual('hello', name)
    self.assertEqual(['world'], value)
    opt, name, value, parser = cb_calls[1]
    self.assertEqual('hello', name)
    self.assertEqual(['world', 'mars'], value)
    opt, name, value, parser = cb_calls[2]
    self.assertEqual('hello', name)
    self.assertEqual([], value)