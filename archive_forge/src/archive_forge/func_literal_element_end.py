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
def literal_element_end(self, name: Tuple[str, str], qname) -> None:
    if name[0]:
        prefix = self._current_context[name[0]]
        if prefix:
            end = '</%s:%s>' % (prefix, name[1])
        else:
            end = '</%s>' % name[1]
    else:
        end = '</%s>' % name[1]
    self.parent.object += self.current.object + end