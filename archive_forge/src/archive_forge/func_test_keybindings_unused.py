import os
import tempfile
import textwrap
import unittest
from bpython import config
def test_keybindings_unused(self):
    struct = self.load_temp_config(textwrap.dedent('\n            [keyboard]\n            help = F4\n            '))
    self.assertEqual(struct.help_key, 'F4')