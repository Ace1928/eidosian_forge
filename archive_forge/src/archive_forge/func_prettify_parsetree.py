from __future__ import annotations
from collections import OrderedDict
from types import MethodType
from typing import (
from pyparsing import ParserElement, ParseResults, TokenConverter, originalTextFor
from rdflib.term import BNode, Identifier, Variable
from rdflib.plugins.sparql.sparql import NotBoundError, SPARQLError  # noqa: E402
def prettify_parsetree(t: ParseResults, indent: str='', depth: int=0) -> str:
    out: List[str] = []
    for e in t.asList():
        out.append(_prettify_sub_parsetree(e, indent, depth + 1))
    for k, v in sorted(t.items()):
        out.append('%s%s- %s:\n' % (indent, '  ' * depth, k))
        out.append(_prettify_sub_parsetree(v, indent, depth + 1))
    return ''.join(out)