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
def pprintAlgebra(q) -> None:

    def pp(p, ind='    '):
        if not isinstance(p, CompValue):
            print(p)
            return
        print('%s(' % (p.name,))
        for k in p:
            print('%s%s =' % (ind, k), end=' ')
            pp(p[k], ind + '    ')
        print('%s)' % ind)
    try:
        pp(q.algebra)
    except AttributeError:
        for x in q:
            pp(x)