from __future__ import annotations
import codecs
import csv
from typing import IO, Dict, List, Optional, Union
from rdflib.plugins.sparql.processor import SPARQLResult
from rdflib.query import Result, ResultParser, ResultSerializer
from rdflib.term import BNode, Identifier, Literal, URIRef, Variable
def serializeTerm(self, term: Optional[Identifier], encoding: str) -> Union[str, Identifier]:
    if term is None:
        return ''
    elif isinstance(term, BNode):
        return f'_:{term}'
    else:
        return term