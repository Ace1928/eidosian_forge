from __future__ import unicode_literals
from xml.etree import ElementTree as ET
from pybtex.database import Entry, Person
from pybtex.database.input import BaseParser
def remove_ns(s):
    if s.startswith(bibtexns):
        return s[len(bibtexns):]