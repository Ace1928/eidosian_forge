import os
import sys
import re
from unittest import TestCase
from .. import Options
from ..CmdLine import parse_command_line
from .Utils import backup_Options, restore_Options, check_global_options
def test_coverage_first(self):
    options, sources = parse_command_line(['--annotate-coverage=my.xml', '--annotate-fullc', 'file3.pyx'])
    self.assertEqual(Options.annotate, 'fullc')
    self.assertEqual(Options.annotate_coverage_xml, 'my.xml')
    self.check_default_global_options(['annotate', 'annotate_coverage_xml'])
    self.check_default_options(options)