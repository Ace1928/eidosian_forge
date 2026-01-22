from __future__ import annotations
from collections import namedtuple
from typing import (
from urllib.parse import urljoin, urlsplit
from rdflib.namespace import RDF
from .errors import (
from .keys import (
from .util import norm_url, source_to_json, split_iri
def find_term(self, idref: str, coercion: Optional[Union[str, Defined]]=None, container: Union[Defined, str]=UNDEF, language: Optional[str]=None, reverse: bool=False):
    lu = self._lookup
    if coercion is None:
        coercion = language
    if coercion is not UNDEF and container:
        found = lu.get((idref, coercion, container, reverse))
        if found:
            return found
    if coercion is not UNDEF:
        found = lu.get((idref, coercion, UNDEF, reverse))
        if found:
            return found
    if container:
        found = lu.get((idref, coercion, container, reverse))
        if found:
            return found
    elif language:
        found = lu.get((idref, UNDEF, LANG, reverse))
        if found:
            return found
    else:
        found = lu.get((idref, coercion or UNDEF, SET, reverse))
        if found:
            return found
    return lu.get((idref, UNDEF, UNDEF, reverse))