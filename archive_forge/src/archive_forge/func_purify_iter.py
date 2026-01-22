from __future__ import absolute_import, unicode_literals
import re
import six
from pybtex.bibtex.exceptions import BibTeXError
from pybtex.py3compat import fix_unicode_literals_in_doctest
from pybtex.utils import pairwise
from pybtex import py3compat
def purify_iter(string):
    for token, brace_level in scan_bibtex_string(string):
        if brace_level == 1 and token.startswith('\\'):
            for char in purify_special_char_re.sub('', token):
                if char.isalnum():
                    yield char
        elif token.isalnum():
            yield token
        elif token.isspace() or token in '-~':
            yield ' '