import re
import sys
import textwrap
from io import StringIO
from .. import commands, export_pot, option, registry, tests
def test_strings_multiline_escapes(self):
    src = 's = "Escaped\\n"\nr = r"Raw\\n"\nt = (\n    "A\\n\\n"\n    "B\\n\\n"\n    "C\\n\\n"\n    )\n'
    _, str_lines = export_pot._parse_source(src)
    if sys.version_info < (3, 8):
        self.expectFailure('Escaped newlines confuses the multiline handling', self.assertNotEqual, str_lines, {'Escaped\n': 0, 'Raw\\n': 2, 'A\n\nB\n\nC\n\n': -2})
    else:
        self.assertEqual(str_lines, {'Escaped\n': 1, 'Raw\\n': 2, 'A\n\nB\n\nC\n\n': 4})