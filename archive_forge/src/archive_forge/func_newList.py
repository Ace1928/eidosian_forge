from __future__ import annotations
import codecs
import os
import re
import sys
import typing
from decimal import Decimal
from typing import (
from uuid import uuid4
from rdflib.compat import long_type
from rdflib.exceptions import ParserError
from rdflib.graph import ConjunctiveGraph, Graph, QuotedGraph
from rdflib.term import (
from rdflib.parser import Parser
def newList(self, n: typing.List[Any], f: Optional[Formula]) -> IdentifiedNode:
    nil = self.newSymbol('http://www.w3.org/1999/02/22-rdf-syntax-ns#nil')
    if not n:
        return nil
    first = self.newSymbol('http://www.w3.org/1999/02/22-rdf-syntax-ns#first')
    rest = self.newSymbol('http://www.w3.org/1999/02/22-rdf-syntax-ns#rest')
    af = a = self.newBlankNode(f)
    for ne in n[:-1]:
        self.makeStatement((f, first, a, ne))
        an = self.newBlankNode(f)
        self.makeStatement((f, rest, a, an))
        a = an
    self.makeStatement((f, first, a, n[-1]))
    self.makeStatement((f, rest, a, nil))
    return af