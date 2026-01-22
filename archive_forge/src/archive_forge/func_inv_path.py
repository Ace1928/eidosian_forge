from __future__ import annotations
import warnings
from functools import total_ordering
from typing import (
from rdflib.term import Node, URIRef
def inv_path(p: Union[URIRef, Path]) -> InvPath:
    """
    inverse path
    """
    return InvPath(p)