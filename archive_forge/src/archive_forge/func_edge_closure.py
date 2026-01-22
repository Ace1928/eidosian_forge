import inspect
import locale
import os
import pydoc
import re
import textwrap
import warnings
from collections import defaultdict, deque
from itertools import chain, combinations, islice, tee
from pprint import pprint
from urllib.request import (
from nltk.collections import *
from nltk.internals import deprecated, raise_unorderable_types, slice_bounds
def edge_closure(tree, children=iter, maxdepth=-1, verbose=False):
    """Yield the edges of a graph in breadth-first order,
    discarding eventual cycles.
    The first argument should be the start node;
    children should be a function taking as argument a graph node
    and returning an iterator of the node's children.

    >>> from nltk.util import edge_closure
    >>> print(list(edge_closure('A', lambda node:{'A':['B','C'], 'B':'C', 'C':'B'}[node])))
    [('A', 'B'), ('A', 'C'), ('B', 'C'), ('C', 'B')]
    """
    traversed = set()
    edges = set()
    queue = deque([(tree, 0)])
    while queue:
        node, depth = queue.popleft()
        traversed.add(node)
        if depth != maxdepth:
            try:
                for child in children(node):
                    if child not in traversed:
                        queue.append((child, depth + 1))
                    elif verbose:
                        warnings.warn(f'Discarded redundant search for {child} at depth {depth + 1}', stacklevel=2)
                    edge = (node, child)
                    if edge not in edges:
                        yield edge
                        edges.add(edge)
            except TypeError:
                pass