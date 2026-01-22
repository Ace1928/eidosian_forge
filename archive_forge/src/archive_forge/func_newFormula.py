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
def newFormula(self) -> Formula:
    fa = getattr(self.graph.store, 'formula_aware', False)
    if not fa:
        raise ParserError('Cannot create formula parser with non-formula-aware store.')
    f = Formula(self.graph)
    return f