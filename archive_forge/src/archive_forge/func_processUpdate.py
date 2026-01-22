from __future__ import annotations
from typing import Any, Mapping, Optional, Union
from rdflib.graph import Graph
from rdflib.plugins.sparql.algebra import translateQuery, translateUpdate
from rdflib.plugins.sparql.evaluate import evalQuery
from rdflib.plugins.sparql.parser import parseQuery, parseUpdate
from rdflib.plugins.sparql.sparql import Query, Update
from rdflib.plugins.sparql.update import evalUpdate
from rdflib.query import Processor, Result, UpdateProcessor
from rdflib.term import Identifier
def processUpdate(graph: Graph, updateString: str, initBindings: Optional[Mapping[str, Identifier]]=None, initNs: Optional[Mapping[str, Any]]=None, base: Optional[str]=None) -> None:
    """
    Process a SPARQL Update Request
    returns Nothing on success or raises Exceptions on error
    """
    evalUpdate(graph, translateUpdate(parseUpdate(updateString), base, initNs), initBindings)