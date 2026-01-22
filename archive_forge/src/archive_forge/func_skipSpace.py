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
def skipSpace(self, argstr: str, i: int) -> int:
    """Skip white space, newlines and comments.
        return -1 if EOF, else position of first non-ws character"""
    try:
        while True:
            ch = argstr[i]
            if ch in {' ', '\t'}:
                i += 1
                continue
            elif ch not in {'#', '\r', '\n'}:
                return i
            break
    except IndexError:
        return -1
    while 1:
        m = eol.match(argstr, i)
        if m is None:
            break
        self.lines += 1
        self.startOfLine = i = m.end()
    m = ws.match(argstr, i)
    if m is not None:
        i = m.end()
    m = eof.match(argstr, i)
    return i if m is None else -1