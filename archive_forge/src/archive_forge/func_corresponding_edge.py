import spherogram
from spherogram.links.tangles import Tangle, OneTangle, MinusOneTangle
import networkx as nx
from random import randint,choice,sample
from spherogram.links.random_links import map_to_link, random_map
def corresponding_edge(new_link, edge):
    G = new_link.dual_graph()
    for new_edge in G.edges:
        if str(new_edge) == str(edge):
            return new_edge