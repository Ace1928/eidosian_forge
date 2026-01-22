import os
import sys
import re
from unittest import TestCase
from .. import Options
from ..CmdLine import parse_command_line
from .Utils import backup_Options, restore_Options, check_global_options
def test_fast_fail(self):
    options, sources = parse_command_line(['--fast-fail', 'source.pyx'])
    self.assertEqual(Options.fast_fail, True)
    self.check_default_global_options(['fast_fail'])
    self.check_default_options(options)