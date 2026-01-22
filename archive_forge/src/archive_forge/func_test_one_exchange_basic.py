import random
import networkx as nx
from networkx.algorithms.approximation import maxcut
def test_one_exchange_basic():
    G = nx.complete_graph(5)
    random.seed(5)
    for u, v, w in G.edges(data=True):
        w['weight'] = random.randrange(-100, 100, 1) / 10
    initial_cut = set(random.sample(sorted(G.nodes()), k=5))
    cut_size, (set1, set2) = maxcut.one_exchange(G, initial_cut, weight='weight', seed=5)
    _is_valid_cut(G, set1, set2)
    _cut_is_locally_optimal(G, cut_size, set1)