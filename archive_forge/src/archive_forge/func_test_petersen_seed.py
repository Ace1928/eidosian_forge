import networkx as nx
from networkx.algorithms.approximation import average_clustering
def test_petersen_seed():
    G = nx.petersen_graph()
    assert average_clustering(G, trials=len(G) // 2, seed=1) == nx.average_clustering(G)