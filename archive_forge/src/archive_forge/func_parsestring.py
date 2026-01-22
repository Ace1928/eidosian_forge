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
def parsestring(self, s: Union[bytes, bytearray, str], **kwargs) -> None:
    """Parse s as an N-Triples string."""
    if not isinstance(s, (str, bytes, bytearray)):
        raise ParseError('Item to parse must be a string instance.')
    f: Union[codecs.StreamReader, StringIO]
    if isinstance(s, (bytes, bytearray)):
        f = codecs.getreader('utf-8')(BytesIO(s))
    else:
        f = StringIO(s)
    self.parse(f, **kwargs)