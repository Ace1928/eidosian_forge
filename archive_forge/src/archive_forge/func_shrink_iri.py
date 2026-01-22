from __future__ import annotations
from collections import namedtuple
from typing import (
from urllib.parse import urljoin, urlsplit
from rdflib.namespace import RDF
from .errors import (
from .keys import (
from .util import norm_url, source_to_json, split_iri
def shrink_iri(self, iri: str) -> str:
    ns, name = split_iri(str(iri))
    pfx = self._prefixes.get(ns)
    if pfx:
        return ':'.join((pfx, name))
    elif self._base:
        if str(iri) == self._base:
            return ''
        elif iri.startswith(self._basedomain):
            return iri[len(self._basedomain):]
    return iri