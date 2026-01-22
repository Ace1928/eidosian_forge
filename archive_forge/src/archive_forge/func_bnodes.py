from __future__ import annotations
import collections
import datetime
import itertools
import typing as t
from collections.abc import Mapping, MutableMapping
from typing import (
import isodate
import rdflib.plugins.sparql
from rdflib.graph import ConjunctiveGraph, Graph
from rdflib.namespace import NamespaceManager
from rdflib.plugins.sparql.parserutils import CompValue
from rdflib.term import BNode, Identifier, Literal, Node, URIRef, Variable
@property
def bnodes(self) -> t.Mapping[Identifier, BNode]:
    return self.ctx.bnodes