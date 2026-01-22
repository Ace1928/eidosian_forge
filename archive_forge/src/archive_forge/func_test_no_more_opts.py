import re
from .. import bzr, commands, controldir, errors, option, registry
from ..builtins import cmd_commit
from ..bzr import knitrepo
from ..commands import parse_args
from . import TestCase
def test_no_more_opts(self):
    """Terminated options"""
    self.assertEqual((['-file-with-dashes'], {'author': [], 'exclude': [], 'fixes': [], 'bugs': []}), parse_args(cmd_commit(), ['--', '-file-with-dashes']))