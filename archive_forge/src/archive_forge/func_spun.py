from .graphs import ReducedGraph, Digraph, Poset
from collections import deque
import operator
def spun(self, start=0):
    """
        Generator for letters in cyclic order, starting at start.
        """
    N = len(self)
    for n in range(start, start + N):
        yield self[n % N]