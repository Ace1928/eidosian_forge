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
def on_cycle(self, node, path):
    logger.debug('Cycle encountered in the SPPF at node: %s. As infinite ambiguities cannot be represented in a tree, this family of derivations will be discarded.', node)
    self._cycle_node = node
    self._on_cycle_retreat = True