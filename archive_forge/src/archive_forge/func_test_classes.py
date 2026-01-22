import re
import sys
import textwrap
from io import StringIO
from .. import commands, export_pot, option, registry, tests
def test_classes(self):
    src = '\nclass Ancient:\n    """Old style class"""\n\nclass Modern(object):\n    """New style class"""\n'
    cls_lines, _ = export_pot._parse_source(src)
    self.assertEqual(cls_lines, {'Ancient': 2, 'Modern': 5})