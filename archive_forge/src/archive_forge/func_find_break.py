from __future__ import absolute_import, unicode_literals
import re
import six
from pybtex.bibtex.exceptions import BibTeXError
from pybtex.py3compat import fix_unicode_literals_in_doctest
from pybtex.utils import pairwise
from pybtex import py3compat
def find_break(string):
    for prev_match, match in pairwise(whitespace_re.finditer(string)):
        if (match is None or match.start() > width) and prev_match.start() > min_width:
            return prev_match.start()