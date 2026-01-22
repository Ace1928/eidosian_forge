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
def get_cycle_in_path(self, node, path):
    """A utility function for use in ``on_cycle`` to obtain a slice of
        ``path`` that only contains the nodes that make up the cycle."""
    index = len(path) - 1
    while id(path[index]) != id(node):
        index -= 1
    return path[index:]