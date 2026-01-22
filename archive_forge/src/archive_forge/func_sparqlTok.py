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
def sparqlTok(self, tok: str, argstr: str, i: int) -> int:
    """Check for SPARQL keyword.  Space must have been stripped on entry
        and we must not be at end of file.
        Case insensitive and not preceded by @
        """
    assert tok[0] not in _notNameChars
    len_tok = len(tok)
    if argstr[i:i + len_tok].lower() == tok.lower() and argstr[i + len_tok] in _notQNameChars:
        i += len_tok
        return i
    else:
        return -1