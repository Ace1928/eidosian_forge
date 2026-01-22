import sys
from copy import deepcopy
from typing import List, Callable, Iterator, Union, Optional, Generic, TypeVar, TYPE_CHECKING
from collections import OrderedDict
def iter_subtrees_topdown(self):
    """Breadth-first iteration.

        Iterates over all the subtrees, return nodes in order like pretty() does.
        """
    stack = [self]
    stack_append = stack.append
    stack_pop = stack.pop
    while stack:
        node = stack_pop()
        if not isinstance(node, Tree):
            continue
        yield node
        for child in reversed(node.children):
            stack_append(child)