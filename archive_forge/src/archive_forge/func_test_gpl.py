import ast
import os
import re
import sys
import breezy.branch
from breezy import osutils
from breezy.tests import TestCase, TestSkipped, features
def test_gpl(self):
    """Test that all .py and .pyx files have a GPL disclaimer."""
    incorrect = []
    gpl_txt = '\n# This program is free software; you can redistribute it and/or modify\n# it under the terms of the GNU General Public License as published by\n# the Free Software Foundation; either version 2 of the License, or\n# (at your option) any later version.\n#\n# This program is distributed in the hope that it will be useful,\n# but WITHOUT ANY WARRANTY; without even the implied warranty of\n# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the\n# GNU General Public License for more details.\n#\n# You should have received a copy of the GNU General Public License\n# along with this program; if not, write to the Free Software\n# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA\n'
    gpl_re = re.compile(re.escape(gpl_txt), re.MULTILINE)
    for fname, text in self.get_source_file_contents(extensions=('.py', '.pyx')):
        if self.is_license_exception(fname):
            continue
        if not gpl_re.search(text):
            incorrect.append(fname)
    if incorrect:
        help_text = ['Some files have missing or incomplete GPL statement', '', 'Please either add them to the list of LICENSE_EXCEPTIONS in breezy/tests/test_source.py', 'Or add the following text to the beginning:', gpl_txt]
        for fname in incorrect:
            help_text.append(' ' * 4 + fname)
        self.fail('\n'.join(help_text))