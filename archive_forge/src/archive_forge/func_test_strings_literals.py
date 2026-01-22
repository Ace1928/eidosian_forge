import re
import sys
import textwrap
from io import StringIO
from .. import commands, export_pot, option, registry, tests
def test_strings_literals(self):
    src = 's = "One"\nt = (2, "Two")\nf = dict(key="Three")\n'
    _, str_lines = export_pot._parse_source(src)
    self.assertEqual(str_lines, {'One': 1, 'Two': 2, 'Three': 3})