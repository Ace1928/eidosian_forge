import random
import networkx as nx
from networkx.algorithms.approximation import maxcut
def test_negative_weights():
    G = nx.complete_graph(5)
    random.seed(5)
    for u, v, w in G.edges(data=True):
        w['weight'] = -1 * random.random()
    initial_cut = set(random.sample(sorted(G.nodes()), k=5))
    cut_size, (set1, set2) = maxcut.one_exchange(G, initial_cut, weight='weight')
    _is_valid_cut(G, set1, set2)
    _cut_is_locally_optimal(G, cut_size, set1)
    assert len(set1) == len(G.nodes) or len(set2) == len(G.nodes)