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
def translateUpdate1(u: CompValue, prologue: Prologue) -> CompValue:
    if u.name in ('Load', 'Clear', 'Drop', 'Create'):
        pass
    elif u.name in ('Add', 'Move', 'Copy'):
        pass
    elif u.name in ('InsertData', 'DeleteData', 'DeleteWhere'):
        t, q = translateQuads(u.quads)
        u['quads'] = q
        u['triples'] = t
        if u.name in ('DeleteWhere', 'DeleteData'):
            pass
    elif u.name == 'Modify':
        if u.delete:
            u.delete['triples'], u.delete['quads'] = translateQuads(u.delete.quads)
        if u.insert:
            u.insert['triples'], u.insert['quads'] = translateQuads(u.insert.quads)
        u['where'] = translateGroupGraphPattern(u.where)
    else:
        raise Exception('Unknown type of update operation: %s' % u)
    u.prologue = prologue
    return u