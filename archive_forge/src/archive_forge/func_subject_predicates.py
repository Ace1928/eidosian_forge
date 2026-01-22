import collections
import re
from typing import (
from rdflib.graph import DATASET_DEFAULT_GRAPH_ID, Graph
from rdflib.plugins.stores.regexmatching import NATIVE_REGEX
from rdflib.store import Store
from rdflib.term import BNode, Identifier, Node, URIRef, Variable
from .sparqlconnector import SPARQLConnector
def subject_predicates(self, object: Optional['_ObjectType']=None) -> Generator[Tuple['_SubjectType', '_PredicateType'], None, None]:
    """A generator of (subject, predicate) tuples for the given object"""
    for t, c in self.triples((None, None, object)):
        yield (t[0], t[1])