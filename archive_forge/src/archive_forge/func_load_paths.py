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
def load_paths(self):
    for transitive, node in self.paths:
        if transitive.next_titem is not None:
            vn = type(self)(transitive.next_titem.s, transitive.next_titem.start, self.end)
            vn.add_path(transitive.next_titem, node)
            self.add_family(transitive.reduction.rule.origin, transitive.reduction.rule, transitive.reduction.start, transitive.reduction.node, vn)
        else:
            self.add_family(transitive.reduction.rule.origin, transitive.reduction.rule, transitive.reduction.start, transitive.reduction.node, node)
    self.paths_loaded = True