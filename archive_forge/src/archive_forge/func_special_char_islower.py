from __future__ import unicode_literals
from __future__ import print_function
import re
import six
import textwrap
from pybtex.exceptions import PybtexError
from pybtex.utils import (
from pybtex.richtext import Text
from pybtex.bibtex.utils import split_tex_string, scan_bibtex_string
from pybtex.errors import report_error
from pybtex.py3compat import fix_unicode_literals_in_doctest, python_2_unicode_compatible
from pybtex.plugin import find_plugin
def special_char_islower(special_char):
    control_sequence = True
    for char in special_char[1:]:
        if control_sequence:
            if not char.isalpha():
                control_sequence = False
        elif char.isalpha():
            return char.islower()
    return False