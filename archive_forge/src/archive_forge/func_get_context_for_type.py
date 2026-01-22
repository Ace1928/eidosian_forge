from __future__ import annotations
from collections import namedtuple
from typing import (
from urllib.parse import urljoin, urlsplit
from rdflib.namespace import RDF
from .errors import (
from .keys import (
from .util import norm_url, source_to_json, split_iri
def get_context_for_type(self, node: Any) -> Optional['Context']:
    if self.version >= 1.1:
        rtype = self.get_type(node) if isinstance(node, dict) else None
        if not isinstance(rtype, list):
            rtype = [rtype] if rtype else []
        for rt in rtype:
            typeterm = self.terms.get(rt)
            if typeterm:
                break
        else:
            typeterm = None
        if typeterm and typeterm.context:
            subcontext = self.subcontext(typeterm.context, propagate=False)
            if subcontext:
                return subcontext
    return self.parent if self.propagate is False else self