from __future__ import unicode_literals
import re
from pybtex.bibtex.utils import bibtex_abbreviate, bibtex_len
from pybtex.database import Person
from pybtex.scanner import (
def check_format_chars(value):
    value = value.lower()
    if format_chars is not None or len(value) not in [1, 2] or value[0] != value[-1] or (value[0] not in 'flvj'):
        raise PybtexSyntaxError(u'name format string "{0}" has illegal brace-level-1 letters: {1}'.format(self.text, token.value), self)