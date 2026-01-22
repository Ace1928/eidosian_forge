import spherogram
from spherogram.links.tangles import Tangle, OneTangle, MinusOneTangle
import networkx as nx
from random import randint,choice,sample
from spherogram.links.random_links import map_to_link, random_map
def random_mutate(link):
    G = link.dual_graph()
    v = choice(tuple(G.vertices))
    all_fcs = tuple(all_four_cycles_at_vertex(G, v))
    if len(all_fcs) == 0:
        return link.copy()
    four_cycle = choice(all_fcs)
    return mutate(link, four_cycle)