from .graphs import ReducedGraph, Digraph, Poset
from collections import deque
import operator
def syllables(self):
    if len(self) == 0:
        return []
    ans, curr = ([], None)
    for x in self:
        g = abs(x)
        e = 1 if x > 0 else -1
        if g == curr:
            count += e
        else:
            if curr is not None:
                ans.append((curr, count))
            curr, count = (g, e)
    ans.append((curr, count))
    return ans