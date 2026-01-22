import spherogram
from spherogram.links.tangles import Tangle, OneTangle, MinusOneTangle
import networkx as nx
from random import randint,choice,sample
from spherogram.links.random_links import map_to_link, random_map
def trace_boundary_component(start_cs, full_boundary):
    boundary_comp = [start_cs]
    cs = start_cs.next_corner()
    i = 0
    while cs != start_cs:
        while cs.rotate(1) not in full_boundary:
            print(cs)
            cs = cs.next_corner()
            i += 1
            if i > 100:
                raise Exception()
        cs = cs.rotate(1)
        boundary_comp.append(cs)
    boundary_comp.pop(-1)
    return boundary_comp