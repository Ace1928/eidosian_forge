from itertools import chain, combinations, permutations, product
import networkx as nx
from networkx import density
from networkx.exception import NetworkXException
from networkx.utils import arbitrary_element
def node_data(b):
    S = G.subgraph(b)
    return {'graph': S, 'nnodes': len(S), 'nedges': S.number_of_edges(), 'density': density(S)}