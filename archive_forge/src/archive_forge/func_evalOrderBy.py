from __future__ import annotations
import collections
import itertools
import json as j
import re
from typing import (
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from pyparsing import ParseException
from rdflib.graph import Graph
from rdflib.plugins.sparql import CUSTOM_EVALS, parser
from rdflib.plugins.sparql.aggregates import Aggregator
from rdflib.plugins.sparql.evalutils import (
from rdflib.plugins.sparql.parserutils import CompValue, value
from rdflib.plugins.sparql.sparql import (
from rdflib.term import BNode, Identifier, Literal, URIRef, Variable
def evalOrderBy(ctx: QueryContext, part: CompValue) -> Generator[FrozenBindings, None, None]:
    res = evalPart(ctx, part.p)
    for e in reversed(part.expr):
        reverse = bool(e.order and e.order == 'DESC')
        res = sorted(res, key=lambda x: _val(value(x, e.expr, variables=True)), reverse=reverse)
    return res