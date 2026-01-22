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
def do_de_skolemize2(t: _TripleType) -> _TripleType:
    s, p, o = t
    if RDFLibGenid._is_rdflib_skolem(s):
        s = RDFLibGenid(s).de_skolemize()
    elif Genid._is_external_skolem(s):
        s = Genid(s).de_skolemize()
    if RDFLibGenid._is_rdflib_skolem(o):
        o = RDFLibGenid(o).de_skolemize()
    elif Genid._is_external_skolem(o):
        o = Genid(o).de_skolemize()
    return (s, p, o)