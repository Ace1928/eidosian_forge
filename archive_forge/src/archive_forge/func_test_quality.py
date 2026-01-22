import networkx as nx
def test_quality():
    G = nx.LFR_benchmark_graph(250, 3, 1.5, 0.009, average_degree=5, min_community=20, seed=10)
    H = nx.gn_graph(200, seed=1234)
    I = nx.MultiGraph(G)
    J = nx.MultiDiGraph(H)
    partition = nx.community.louvain_communities(G)
    partition2 = nx.community.louvain_communities(H)
    partition3 = nx.community.louvain_communities(I)
    partition4 = nx.community.louvain_communities(J)
    quality = nx.community.partition_quality(G, partition)[0]
    quality2 = nx.community.partition_quality(H, partition2)[0]
    quality3 = nx.community.partition_quality(I, partition3)[0]
    quality4 = nx.community.partition_quality(J, partition4)[0]
    assert quality >= 0.65
    assert quality2 >= 0.65
    assert quality3 >= 0.65
    assert quality4 >= 0.65