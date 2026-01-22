import warnings
from collections import Counter, defaultdict
from math import comb, factorial
import networkx as nx
from networkx.utils import py_random_state
def get_children(parent, paths):
    children = defaultdict(list)
    for path in paths:
        if not path:
            tree.add_edge(parent, NIL)
            continue
        child, *rest = path
        children[child].append(rest)
    return children