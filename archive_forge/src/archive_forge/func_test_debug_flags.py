import os
import sys
import re
from unittest import TestCase
from .. import Options
from ..CmdLine import parse_command_line
from .Utils import backup_Options, restore_Options, check_global_options
def test_debug_flags(self):
    options, sources = parse_command_line(['--debug-disposal-code', '--debug-coercion', 'file3.pyx'])
    from Cython.Compiler import DebugFlags
    for name in ['debug_disposal_code', 'debug_temp_alloc', 'debug_coercion']:
        self.assertEqual(getattr(DebugFlags, name), name in ['debug_disposal_code', 'debug_coercion'])
        setattr(DebugFlags, name, 0)