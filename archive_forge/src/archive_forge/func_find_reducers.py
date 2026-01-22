from .graphs import ReducedGraph, Digraph, Poset
from collections import deque
import operator
def find_reducers(self):
    whitehead = self.whitehead_graph()
    reducers = []
    levels = []
    for x in self.generators:
        cut = whitehead.one_min_cut(x, -x)
        valence = whitehead.multi_valence(x)
        length_change = cut['size'] - valence
        if length_change < 0:
            reducers.append((length_change, x, cut['set']))
        elif length_change == 0:
            levels.append((x, cut))
    reducers.sort(key=lambda x: x[0])
    return (reducers, levels)