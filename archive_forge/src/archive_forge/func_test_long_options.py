import os
import sys
import re
from unittest import TestCase
from .. import Options
from ..CmdLine import parse_command_line
from .Utils import backup_Options, restore_Options, check_global_options
def test_long_options(self):
    options, sources = parse_command_line(['--version', '--create-listing', '--cplus', '--embed', '--timestamps', '--verbose', '--verbose', '--verbose', '--embed-positions', '--no-docstrings', '--annotate', '--lenient'])
    self.assertFalse(sources)
    self.assertTrue(options.show_version)
    self.assertTrue(options.use_listing_file)
    self.assertTrue(options.cplus)
    self.assertEqual(Options.embed, 'main')
    self.assertTrue(options.timestamps)
    self.assertTrue(options.verbose >= 3)
    self.assertTrue(Options.embed_pos_in_docstring)
    self.assertFalse(Options.docstrings)
    self.assertTrue(Options.annotate)
    self.assertFalse(Options.error_on_unknown_names)
    self.assertFalse(Options.error_on_uninitialized)
    options, sources = parse_command_line(['--force', 'source.pyx'])
    self.assertTrue(sources)
    self.assertTrue(len(sources) == 1)
    self.assertFalse(options.timestamps)