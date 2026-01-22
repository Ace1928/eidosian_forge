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
def parse_string_body(self, body_end):
    self.current_field_name = self.required([self.NAME]).value
    self.required([self.EQUALS])
    self.parse_value()
    self.macros[self.current_field_name] = ''.join(self.current_value)