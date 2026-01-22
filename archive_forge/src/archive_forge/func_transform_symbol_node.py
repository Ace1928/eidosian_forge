from typing import Type, AbstractSet
from random import randint
from collections import deque
from operator import attrgetter
from importlib import import_module
from functools import partial
from ..parse_tree_builder import AmbiguousIntermediateExpander
from ..visitors import Discard
from ..utils import logger, OrderedSet
from ..tree import Tree
def transform_symbol_node(self, node, data):
    if id(node) not in self._successful_visits:
        return Discard
    r = self._check_cycle(node)
    if r is Discard:
        return r
    self._successful_visits.remove(id(node))
    data = self._collapse_ambig(data)
    return self._call_ambig_func(node, data)