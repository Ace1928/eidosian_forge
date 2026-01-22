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
def parse_value_part(self):
    token = self.required([self.QUOTE, self.LBRACE, self.NUMBER, self.NAME], description='field value')
    if token.pattern is self.QUOTE:
        value_part = self.flatten_string(self.parse_string(string_end=self.QUOTE))
    elif token.pattern is self.LBRACE:
        value_part = self.flatten_string(self.parse_string(string_end=self.RBRACE))
    elif token.pattern is self.NUMBER:
        value_part = token.value
    else:
        value_part = self.substitute_macro(token.value)
    return value_part