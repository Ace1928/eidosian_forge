from math import sqrt
import networkx as nx
from networkx.utils import py_random_state
def uncompact(creation_sequence):
    """
    Converts a compact creation sequence for a threshold
    graph to a standard creation sequence (unlabeled).
    If the creation_sequence is already standard, return it.
    See creation_sequence.
    """
    first = creation_sequence[0]
    if isinstance(first, str):
        return creation_sequence
    elif isinstance(first, tuple):
        return creation_sequence
    elif isinstance(first, int):
        ccscopy = creation_sequence[:]
    else:
        raise TypeError('Not a valid creation sequence type')
    cs = []
    while ccscopy:
        cs.extend(ccscopy.pop(0) * ['d'])
        if ccscopy:
            cs.extend(ccscopy.pop(0) * ['i'])
    return cs