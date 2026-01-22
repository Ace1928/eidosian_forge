from __future__ import annotations
from typing import TYPE_CHECKING, Any, Dict, List, NoReturn, Optional, Tuple
from urllib.parse import urldefrag, urljoin
from xml.sax import handler, make_parser, xmlreader
from xml.sax.handler import ErrorHandler
from xml.sax.saxutils import escape, quoteattr
from rdflib.exceptions import Error, ParserError
from rdflib.graph import Graph
from rdflib.namespace import RDF, is_ncname
from rdflib.parser import InputSource, Parser
from rdflib.plugins.parsers.RDFVOC import RDFVOC
from rdflib.term import BNode, Identifier, Literal, URIRef
def literal_element_start(self, name: Tuple[str, str], qname, attrs: AttributesImpl) -> None:
    current = self.current
    self.next.start = self.literal_element_start
    self.next.char = self.literal_element_char
    self.next.end = self.literal_element_end
    current.declared = self.parent.declared.copy()
    if name[0]:
        prefix = self._current_context[name[0]]
        if prefix:
            current.object = '<%s:%s' % (prefix, name[1])
        else:
            current.object = '<%s' % name[1]
        if not name[0] in current.declared:
            current.declared[name[0]] = prefix
            if prefix:
                current.object += ' xmlns:%s="%s"' % (prefix, name[0])
            else:
                current.object += ' xmlns="%s"' % name[0]
    else:
        current.object = '<%s' % name[1]
    for name, value in attrs.items():
        if name[0]:
            if not name[0] in current.declared:
                current.declared[name[0]] = self._current_context[name[0]]
            name = current.declared[name[0]] + ':' + name[1]
        else:
            name = name[1]
        current.object += ' %s=%s' % (name, quoteattr(value))
    current.object += '>'