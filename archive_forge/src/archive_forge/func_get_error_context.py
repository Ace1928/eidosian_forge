from __future__ import unicode_literals
import re
from string import ascii_letters, digits
import six
from pybtex import textutils
from pybtex.bibtex.utils import split_name_list
from pybtex.database import Entry, Person, BibliographyDataError
from pybtex.database.input import BaseParser
from pybtex.scanner import (
from pybtex.utils import CaseInsensitiveDict, CaseInsensitiveSet
def get_error_context(self, context_info):
    error_start, lineno, error_pos = context_info
    before_error = self.text[error_start:error_pos]
    if not before_error.endswith('\n'):
        eol = self.NEWLINE.search(self.text, error_pos)
        error_end = eol.end() if eol else self.end_pos
    else:
        error_end = error_pos
    context = self.text[error_start:error_end].rstrip('\r\n')
    colno = len(before_error.splitlines()[-1])
    return (context, lineno, colno)