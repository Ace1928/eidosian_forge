import os
import sys
import re
from unittest import TestCase
from .. import Options
from ..CmdLine import parse_command_line
from .Utils import backup_Options, restore_Options, check_global_options
def test_short_w(self):
    options, sources = parse_command_line(['-w', 'my_working_path', 'source.pyx'])
    self.assertEqual(options.working_path, 'my_working_path')
    self.check_default_global_options()
    self.check_default_options(options, ['working_path'])