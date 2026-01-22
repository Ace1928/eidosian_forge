from __future__ import annotations
import codecs
import re
from io import BytesIO, StringIO, TextIOBase
from typing import (
from rdflib.compat import _string_escape_map, decodeUnicodeEscape
from rdflib.exceptions import ParserError as ParseError
from rdflib.parser import InputSource, Parser
from rdflib.term import BNode as bNode
from rdflib.term import Literal
from rdflib.term import URIRef
from rdflib.term import URIRef as URI
def eat(self, pattern: Pattern[str]) -> Match[str]:
    m = pattern.match(self.line)
    if not m:
        raise ParseError('Failed to eat %s at %s' % (pattern.pattern, self.line))
    self.line = self.line[m.end():]
    return m