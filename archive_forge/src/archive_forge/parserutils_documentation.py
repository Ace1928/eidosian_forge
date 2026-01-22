from __future__ import annotations
from collections import OrderedDict
from types import MethodType
from typing import (
from pyparsing import ParserElement, ParseResults, TokenConverter, originalTextFor
from rdflib.term import BNode, Identifier, Variable
from rdflib.plugins.sparql.sparql import NotBoundError, SPARQLError  # noqa: E402

    A pyparsing token for grouping together things with a label
    Any sub-tokens that are not Params will be ignored.

    Returns CompValue / Expr objects - depending on whether evalFn is set.
    