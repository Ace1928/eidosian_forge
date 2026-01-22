from itertools import chain, combinations, permutations, product
import networkx as nx
from networkx import density
from networkx.exception import NetworkXException
from networkx.utils import arbitrary_element
def edge_data(b, c):
    edgedata = (d for u, v, d in G.edges(b | c, data=True) if u in b and v in c or (u in c and v in b))
    return {'weight': sum((d.get(weight, 1) for d in edgedata))}