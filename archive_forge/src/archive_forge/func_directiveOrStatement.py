from __future__ import annotations
from typing import Any, MutableSequence
from rdflib.graph import ConjunctiveGraph, Graph
from rdflib.parser import InputSource, Parser
from .notation3 import RDFSink, SinkParser
def directiveOrStatement(self, argstr: str, h: int) -> int:
    i = self.skipSpace(argstr, h)
    if i < 0:
        return i
    j = self.graph(argstr, i)
    if j >= 0:
        return j
    j = self.sparqlDirective(argstr, i)
    if j >= 0:
        return j
    j = self.directive(argstr, i)
    if j >= 0:
        return self.checkDot(argstr, j)
    j = self.statement(argstr, i)
    if j >= 0:
        return self.checkDot(argstr, j)
    return j