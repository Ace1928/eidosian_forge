import re
import sys
import textwrap
from io import StringIO
from .. import commands, export_pot, option, registry, tests
def test_strings_docstrings(self):
    src = '"""Module"""\n\ndef function():\n    """Function"""\n\nclass Class(object):\n    """Class"""\n\n    def method(self):\n        """Method"""\n'
    _, str_lines = export_pot._parse_source(src)
    self.assertEqual(str_lines, {'Module': 1, 'Function': 4, 'Class': 7, 'Method': 10})