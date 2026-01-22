from __future__ import annotations
from collections import namedtuple
from typing import (
from urllib.parse import urljoin, urlsplit
from rdflib.namespace import RDF
from .errors import (
from .keys import (
from .util import norm_url, source_to_json, split_iri
def subcontext(self, source: Any, propagate: bool=True) -> 'Context':
    parent = self.parent if self.propagate is False else self
    return parent._subcontext(source, propagate)