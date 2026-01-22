import collections
import re
from typing import (
from rdflib.graph import DATASET_DEFAULT_GRAPH_ID, Graph
from rdflib.plugins.stores.regexmatching import NATIVE_REGEX
from rdflib.store import Store
from rdflib.term import BNode, Identifier, Node, URIRef, Variable
from .sparqlconnector import SPARQLConnector
def subject_objects(self, predicate: Optional['_PredicateType']=None) -> Generator[Tuple['_SubjectType', '_ObjectType'], None, None]:
    """A generator of (subject, object) tuples for the given predicate"""
    for t, c in self.triples((None, predicate, None)):
        yield (t[0], t[2])