import networkx
import random
from .links import Strand
from ..graphs import CyclicList, Digraph
from collections import namedtuple, Counter
def subdivide_edge(crossing_strand, n):
    """
    Given a CrossingStrand, subdivides the edge for which it's the *head*
    into (n + 1) pieces.

    WARNING: this breaks several of the link's internal data structures.
    """
    head = crossing_strand
    backwards = not head in head.crossing.entry_points()
    if backwards:
        head = head.opposite()
    tail = head.opposite()
    strands = [Strand() for i in range(n)]
    strands[0][0] = tail.crossing[tail.strand_index]
    for i in range(n - 1):
        strands[i][1] = strands[i + 1][0]
    strands[-1][1] = head.crossing[head.strand_index]