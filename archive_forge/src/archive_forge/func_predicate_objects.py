import collections
import re
from typing import (
from rdflib.graph import DATASET_DEFAULT_GRAPH_ID, Graph
from rdflib.plugins.stores.regexmatching import NATIVE_REGEX
from rdflib.store import Store
from rdflib.term import BNode, Identifier, Node, URIRef, Variable
from .sparqlconnector import SPARQLConnector
def predicate_objects(self, subject: Optional['_SubjectType']=None) -> Generator[Tuple['_PredicateType', '_ObjectType'], None, None]:
    """A generator of (predicate, object) tuples for the given subject"""
    for t, c in self.triples((subject, None, None)):
        yield (t[1], t[2])