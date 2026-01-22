import os
import sys
import re
from unittest import TestCase
from .. import Options
from ..CmdLine import parse_command_line
from .Utils import backup_Options, restore_Options, check_global_options
def test_two_embeds(self):
    options, sources = parse_command_line(['--embed', '--embed=huhu', 'source.pyx'])
    self.assertEqual(sources, ['source.pyx'])
    self.assertEqual(Options.embed, 'huhu')