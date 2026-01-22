import networkx
import random
from .links import Strand
from ..graphs import CyclicList, Digraph
from collections import namedtuple, Counter
def break_into_arrows(self):
    arrows = []
    for s in self.strand_CEPs:
        arrow = [s, s.next()]
        while not isinstance(arrow[-1].crossing, Strand):
            arrow.append(arrow[-1].next())
        arrows.append(arrow)
    undercrossings = {}
    for i, arrow in enumerate(arrows):
        for a in arrow[1:-1]:
            if a.is_under_crossing():
                undercrossings[a] = i
    crossings = []
    for i, arrow in enumerate(arrows):
        for a in arrow[1:-1]:
            if a.is_over_crossing():
                crossings.append((undercrossings[a.other()], i, False, a.crossing.label))
    return (arrows, crossings)