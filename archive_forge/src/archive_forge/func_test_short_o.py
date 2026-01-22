import os
import sys
import re
from unittest import TestCase
from .. import Options
from ..CmdLine import parse_command_line
from .Utils import backup_Options, restore_Options, check_global_options
def test_short_o(self):
    options, sources = parse_command_line(['-o', 'my_output', 'source.pyx'])
    self.assertEqual(options.output_file, 'my_output')
    self.check_default_global_options()
    self.check_default_options(options, ['output_file'])