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
def convert_node_arg(self, node_arg: typing.Union[Identifier, CompValue, Expr, str]) -> str:
    if isinstance(node_arg, Identifier):
        if node_arg in self.aggr_vars.keys():
            grp_var = self.aggr_vars[node_arg].pop(0).n3()
            return grp_var
        else:
            return node_arg.n3()
    elif isinstance(node_arg, CompValue):
        return '{' + node_arg.name + '}'
    elif isinstance(node_arg, Expr):
        return '{' + node_arg.name + '}'
    elif isinstance(node_arg, str):
        return node_arg
    else:
        raise ExpressionNotCoveredException('The expression {0} might not be covered yet.'.format(node_arg))