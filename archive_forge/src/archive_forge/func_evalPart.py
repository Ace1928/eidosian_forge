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
def evalPart(ctx: QueryContext, part: CompValue) -> Any:
    for name, c in CUSTOM_EVALS.items():
        try:
            return c(ctx, part)
        except NotImplementedError:
            pass
    if part.name == 'BGP':
        triples = sorted(part.triples, key=lambda t: len([n for n in t if ctx[n] is None]))
        return evalBGP(ctx, triples)
    elif part.name == 'Filter':
        return evalFilter(ctx, part)
    elif part.name == 'Join':
        return evalJoin(ctx, part)
    elif part.name == 'LeftJoin':
        return evalLeftJoin(ctx, part)
    elif part.name == 'Graph':
        return evalGraph(ctx, part)
    elif part.name == 'Union':
        return evalUnion(ctx, part)
    elif part.name == 'ToMultiSet':
        return evalMultiset(ctx, part)
    elif part.name == 'Extend':
        return evalExtend(ctx, part)
    elif part.name == 'Minus':
        return evalMinus(ctx, part)
    elif part.name == 'Project':
        return evalProject(ctx, part)
    elif part.name == 'Slice':
        return evalSlice(ctx, part)
    elif part.name == 'Distinct':
        return evalDistinct(ctx, part)
    elif part.name == 'Reduced':
        return evalReduced(ctx, part)
    elif part.name == 'OrderBy':
        return evalOrderBy(ctx, part)
    elif part.name == 'Group':
        return evalGroup(ctx, part)
    elif part.name == 'AggregateJoin':
        return evalAggregateJoin(ctx, part)
    elif part.name == 'SelectQuery':
        return evalSelectQuery(ctx, part)
    elif part.name == 'AskQuery':
        return evalAskQuery(ctx, part)
    elif part.name == 'ConstructQuery':
        return evalConstructQuery(ctx, part)
    elif part.name == 'ServiceGraphPattern':
        return evalServiceQuery(ctx, part)
    elif part.name == 'DescribeQuery':
        return evalDescribeQuery(ctx, part)
    else:
        raise Exception('I dont know: %s' % part.name)