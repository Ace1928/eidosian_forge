import os
import sys
import re
from unittest import TestCase
from .. import Options
from ..CmdLine import parse_command_line
from .Utils import backup_Options, restore_Options, check_global_options
def test_Wextra(self):
    options, sources = parse_command_line(['-Wextra', 'source.pyx'])
    self.assertEqual(options.compiler_directives, Options.extra_warnings)
    self.check_default_global_options()
    self.check_default_options(options, ['compiler_directives'])