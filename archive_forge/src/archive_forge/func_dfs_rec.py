import collections
import functools
import sys
from numba.core.ir import Loc
from numba.core.errors import UnsupportedError
from numba.core.utils import PYVERSION
def dfs_rec(node):
    if node not in seen:
        seen.add(node)
        stack.append((post_order.append, node))
        for dest in succs[node]:
            if (node, dest) not in back_edges:
                stack.append((dfs_rec, dest))