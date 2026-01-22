import re
from .. import bzr, commands, controldir, errors, option, registry
from ..builtins import cmd_commit
from ..bzr import knitrepo
from ..commands import parse_args
from . import TestCase
def test_allow_dash(self):
    """Test that we can pass a plain '-' as an argument."""
    self.assertEqual(['-'], parse_args(cmd_commit(), ['-'])[0])