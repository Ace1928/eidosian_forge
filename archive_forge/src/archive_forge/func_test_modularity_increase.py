import networkx as nx
def test_modularity_increase():
    G = nx.LFR_benchmark_graph(250, 3, 1.5, 0.009, average_degree=5, min_community=20, seed=10)
    partition = [{u} for u in G.nodes()]
    mod = nx.community.modularity(G, partition)
    partition = nx.community.louvain_communities(G)
    assert nx.community.modularity(G, partition) > mod