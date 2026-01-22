from __future__ import annotations
import warnings
from functools import total_ordering
from typing import (
from rdflib.term import Node, URIRef
def neg_path(p: Union[URIRef, AlternativePath, InvPath]) -> NegatedPath:
    """
    negated path
    """
    return NegatedPath(p)