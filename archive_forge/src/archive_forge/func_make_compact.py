from math import sqrt
import networkx as nx
from networkx.utils import py_random_state
def make_compact(creation_sequence):
    """
    Returns the creation sequence in a compact form
    that is the number of 'i's and 'd's alternating.

    Examples
    --------
    >>> from networkx.algorithms.threshold import make_compact
    >>> make_compact(["d", "i", "i", "d", "d", "i", "i", "i"])
    [1, 2, 2, 3]
    >>> make_compact(["d", "d", "d", "i", "d", "d"])
    [3, 1, 2]

    Notice that the first number is the first vertex
    to be used for construction and so is always 'd'.

    Labeled creation sequences lose their labels in the
    compact representation.

    >>> make_compact([3, 1, 2])
    [3, 1, 2]
    """
    first = creation_sequence[0]
    if isinstance(first, str):
        cs = creation_sequence[:]
    elif isinstance(first, tuple):
        cs = [s[1] for s in creation_sequence]
    elif isinstance(first, int):
        return creation_sequence
    else:
        raise TypeError('Not a valid creation sequence type')
    ccs = []
    count = 1
    for i in range(1, len(cs)):
        if cs[i] == cs[i - 1]:
            count += 1
        else:
            ccs.append(count)
            count = 1
    ccs.append(count)
    return ccs