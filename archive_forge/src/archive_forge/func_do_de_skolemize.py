from __future__ import annotations
import logging
import pathlib
import random
from io import BytesIO
from typing import (
from urllib.parse import urlparse
from urllib.request import url2pathname
import rdflib.exceptions as exceptions
import rdflib.namespace as namespace  # noqa: F401 # This is here because it is used in a docstring.
import rdflib.plugin as plugin
import rdflib.query as query
import rdflib.util  # avoid circular dependency
from rdflib.collection import Collection
from rdflib.exceptions import ParserError
from rdflib.namespace import RDF, Namespace, NamespaceManager
from rdflib.parser import InputSource, Parser, create_input_source
from rdflib.paths import Path
from rdflib.resource import Resource
from rdflib.serializer import Serializer
from rdflib.store import Store
from rdflib.term import (
def do_de_skolemize(uriref: URIRef, t: _TripleType) -> _TripleType:
    s, p, o = t
    if s == uriref:
        if TYPE_CHECKING:
            assert isinstance(s, URIRef)
        s = s.de_skolemize()
    if o == uriref:
        if TYPE_CHECKING:
            assert isinstance(o, URIRef)
        o = o.de_skolemize()
    return (s, p, o)