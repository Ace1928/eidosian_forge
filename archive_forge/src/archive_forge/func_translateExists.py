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
def translateExists(e: typing.Union[Expr, Literal, Variable, URIRef]) -> typing.Union[Expr, Literal, Variable, URIRef]:
    """
    Translate the graph pattern used by EXISTS and NOT EXISTS
    http://www.w3.org/TR/sparql11-query/#sparqlCollectFilters
    """

    def _c(n):
        if isinstance(n, CompValue):
            if n.name in ('Builtin_EXISTS', 'Builtin_NOTEXISTS'):
                n.graph = translateGroupGraphPattern(n.graph)
                if n.graph.name == 'Filter':
                    n.graph.no_isolated_scope = True
    e = traverse(e, visitPost=_c)
    return e