import re
import sys
import textwrap
from io import StringIO
from .. import commands, export_pot, option, registry, tests
def test_option_without_help(self):
    opt = option.Option('helpless')
    self.assertEqual('', self.pot_from_option(opt))