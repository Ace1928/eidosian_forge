import os
import sys
import re
from unittest import TestCase
from .. import Options
from ..CmdLine import parse_command_line
from .Utils import backup_Options, restore_Options, check_global_options
def test_annotate_first_fullc_second(self):
    options, sources = parse_command_line(['--annotate', '--annotate-fullc', 'file3.pyx'])
    self.assertEqual(Options.annotate, 'fullc')
    self.check_default_global_options(['annotate'])
    self.check_default_options(options)