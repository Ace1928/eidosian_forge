from operator import itemgetter
import networkx as nx
def heaviest(G):
    return max(G.edges(data='weight'), key=itemgetter(2))[:2]