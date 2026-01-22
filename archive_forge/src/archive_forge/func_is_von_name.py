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
def is_von_name(string):
    if string[0].isupper():
        return False
    if string[0].islower():
        return True
    else:
        for char, brace_level in scan_bibtex_string(string):
            if brace_level == 0 and char.isalpha():
                return char.islower()
            elif brace_level == 1 and char.startswith('\\'):
                return special_char_islower(char)
    return False