from sympy.core import Basic
from sympy.core.containers import Tuple
from sympy.tensor.array import Array
from sympy.core.sympify import _sympify
from sympy.utilities.iterables import flatten, iterable
from sympy.utilities.misc import as_int
from collections import defaultdict
@staticmethod
def to_prufer(tree, n):
    """Return the Prufer sequence for a tree given as a list of edges where
        ``n`` is the number of nodes in the tree.

        Examples
        ========

        >>> from sympy.combinatorics.prufer import Prufer
        >>> a = Prufer([[0, 1], [0, 2], [0, 3]])
        >>> a.prufer_repr
        [0, 0]
        >>> Prufer.to_prufer([[0, 1], [0, 2], [0, 3]], 4)
        [0, 0]

        See Also
        ========
        prufer_repr: returns Prufer sequence of a Prufer object.

        """
    d = defaultdict(int)
    L = []
    for edge in tree:
        d[edge[0]] += 1
        d[edge[1]] += 1
    for i in range(n - 2):
        for x in range(n):
            if d[x] == 1:
                break
        y = None
        for edge in tree:
            if x == edge[0]:
                y = edge[1]
            elif x == edge[1]:
                y = edge[0]
            if y is not None:
                break
        L.append(y)
        for j in (x, y):
            d[j] -= 1
            if not d[j]:
                d.pop(j)
        tree.remove(edge)
    return L