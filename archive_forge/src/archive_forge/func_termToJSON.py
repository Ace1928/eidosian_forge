from __future__ import annotations
import json
from typing import IO, Any, Dict, Mapping, MutableSequence, Optional
from rdflib.query import Result, ResultException, ResultParser, ResultSerializer
from rdflib.term import BNode, Identifier, Literal, URIRef, Variable
def termToJSON(self: JSONResultSerializer, term: Optional[Identifier]) -> Optional[Dict[str, str]]:
    if isinstance(term, URIRef):
        return {'type': 'uri', 'value': str(term)}
    elif isinstance(term, Literal):
        r = {'type': 'literal', 'value': str(term)}
        if term.datatype is not None:
            r['datatype'] = str(term.datatype)
        if term.language is not None:
            r['xml:lang'] = term.language
        return r
    elif isinstance(term, BNode):
        return {'type': 'bnode', 'value': str(term)}
    elif term is None:
        return None
    else:
        raise ResultException('Unknown term type: %s (%s)' % (term, type(term)))