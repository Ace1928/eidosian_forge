from __future__ import annotations
import codecs
import csv
from typing import IO, Dict, List, Optional, Union
from rdflib.plugins.sparql.processor import SPARQLResult
from rdflib.query import Result, ResultParser, ResultSerializer
from rdflib.term import BNode, Identifier, Literal, URIRef, Variable
def convertTerm(self, t: str) -> Optional[Union[BNode, URIRef, Literal]]:
    if t == '':
        return None
    if t.startswith('_:'):
        return BNode(t)
    if t.startswith('http://') or t.startswith('https://'):
        return URIRef(t)
    return Literal(t)