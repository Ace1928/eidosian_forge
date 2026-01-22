from __future__ import annotations
import collections
import functools
import operator
import typing
from functools import reduce
from typing import (
from pyparsing import ParseResults
from rdflib.paths import (
from rdflib.plugins.sparql.operators import TrueFilter, and_
from rdflib.plugins.sparql.operators import simplify as simplifyFilters
from rdflib.plugins.sparql.parserutils import CompValue, Expr
from rdflib.plugins.sparql.sparql import Prologue, Query, Update
from rdflib.term import BNode, Identifier, Literal, URIRef, Variable
def translateQuery(q: ParseResults, base: Optional[str]=None, initNs: Optional[Mapping[str, Any]]=None) -> Query:
    """
    Translate a query-parsetree to a SPARQL Algebra Expression

    Return a rdflib.plugins.sparql.sparql.Query object
    """
    prologue = translatePrologue(q[0], base, initNs)
    q[1] = traverse(q[1], visitPost=functools.partial(translatePName, prologue=prologue))
    P, PV = translate(q[1])
    datasetClause = q[1].datasetClause
    if q[1].name == 'ConstructQuery':
        template = triples(q[1].template) if q[1].template else None
        res = CompValue(q[1].name, p=P, template=template, datasetClause=datasetClause)
    else:
        res = CompValue(q[1].name, p=P, datasetClause=datasetClause, PV=PV)
    res = traverse(res, visitPost=simplify)
    _traverseAgg(res, visitor=analyse)
    _traverseAgg(res, _addVars)
    return Query(prologue, res)