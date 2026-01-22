from math import sqrt
import networkx as nx
from networkx.utils import py_random_state
def creation_sequence_to_weights(creation_sequence):
    """
    Returns a list of node weights which create the threshold
    graph designated by the creation sequence.  The weights
    are scaled so that the threshold is 1.0.  The order of the
    nodes is the same as that in the creation sequence.
    """
    first = creation_sequence[0]
    if isinstance(first, str):
        if isinstance(creation_sequence, list):
            wseq = creation_sequence[:]
        else:
            wseq = list(creation_sequence)
    elif isinstance(first, tuple):
        wseq = [v[1] for v in creation_sequence]
    elif isinstance(first, int):
        wseq = uncompact(creation_sequence)
    else:
        raise TypeError('Not a valid creation sequence type')
    wseq.reverse()
    w = 0
    prev = 'i'
    for j, s in enumerate(wseq):
        if s == 'i':
            wseq[j] = w
            prev = s
        elif prev == 'i':
            prev = s
            w += 1
    wseq.reverse()
    for j, s in enumerate(wseq):
        if s == 'd':
            wseq[j] = w
            prev = s
        elif prev == 'd':
            prev = s
            w += 1
    if prev == 'd':
        w += 1
    wscale = 1 / w
    return [ww * wscale for ww in wseq]