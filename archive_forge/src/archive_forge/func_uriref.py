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
def uriref(self) -> Union['te.Literal[False]', URI]:
    if self.peek('<'):
        uri = self.eat(r_uriref).group(1)
        uri = unquote(uri)
        uri = uriquote(uri)
        return URI(uri)
    return False