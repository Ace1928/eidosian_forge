import pytest
import networkx as nx
from networkx.algorithms.isomorphism.isomorph import graph_could_be_isomorphic
def test_properties_named_small_graphs(self):
    G = nx.bull_graph()
    assert sorted(G) == list(range(5))
    assert G.number_of_edges() == 5
    assert sorted((d for n, d in G.degree())) == [1, 1, 2, 3, 3]
    assert nx.diameter(G) == 3
    assert nx.radius(G) == 2
    G = nx.chvatal_graph()
    assert sorted(G) == list(range(12))
    assert G.number_of_edges() == 24
    assert [d for n, d in G.degree()] == 12 * [4]
    assert nx.diameter(G) == 2
    assert nx.radius(G) == 2
    G = nx.cubical_graph()
    assert sorted(G) == list(range(8))
    assert G.number_of_edges() == 12
    assert [d for n, d in G.degree()] == 8 * [3]
    assert nx.diameter(G) == 3
    assert nx.radius(G) == 3
    G = nx.desargues_graph()
    assert sorted(G) == list(range(20))
    assert G.number_of_edges() == 30
    assert [d for n, d in G.degree()] == 20 * [3]
    G = nx.diamond_graph()
    assert sorted(G) == list(range(4))
    assert sorted((d for n, d in G.degree())) == [2, 2, 3, 3]
    assert nx.diameter(G) == 2
    assert nx.radius(G) == 1
    G = nx.dodecahedral_graph()
    assert sorted(G) == list(range(20))
    assert G.number_of_edges() == 30
    assert [d for n, d in G.degree()] == 20 * [3]
    assert nx.diameter(G) == 5
    assert nx.radius(G) == 5
    G = nx.frucht_graph()
    assert sorted(G) == list(range(12))
    assert G.number_of_edges() == 18
    assert [d for n, d in G.degree()] == 12 * [3]
    assert nx.diameter(G) == 4
    assert nx.radius(G) == 3
    G = nx.heawood_graph()
    assert sorted(G) == list(range(14))
    assert G.number_of_edges() == 21
    assert [d for n, d in G.degree()] == 14 * [3]
    assert nx.diameter(G) == 3
    assert nx.radius(G) == 3
    G = nx.hoffman_singleton_graph()
    assert sorted(G) == list(range(50))
    assert G.number_of_edges() == 175
    assert [d for n, d in G.degree()] == 50 * [7]
    assert nx.diameter(G) == 2
    assert nx.radius(G) == 2
    G = nx.house_graph()
    assert sorted(G) == list(range(5))
    assert G.number_of_edges() == 6
    assert sorted((d for n, d in G.degree())) == [2, 2, 2, 3, 3]
    assert nx.diameter(G) == 2
    assert nx.radius(G) == 2
    G = nx.house_x_graph()
    assert sorted(G) == list(range(5))
    assert G.number_of_edges() == 8
    assert sorted((d for n, d in G.degree())) == [2, 3, 3, 4, 4]
    assert nx.diameter(G) == 2
    assert nx.radius(G) == 1
    G = nx.icosahedral_graph()
    assert sorted(G) == list(range(12))
    assert G.number_of_edges() == 30
    assert [d for n, d in G.degree()] == [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5]
    assert nx.diameter(G) == 3
    assert nx.radius(G) == 3
    G = nx.krackhardt_kite_graph()
    assert sorted(G) == list(range(10))
    assert G.number_of_edges() == 18
    assert sorted((d for n, d in G.degree())) == [1, 2, 3, 3, 3, 4, 4, 5, 5, 6]
    G = nx.moebius_kantor_graph()
    assert sorted(G) == list(range(16))
    assert G.number_of_edges() == 24
    assert [d for n, d in G.degree()] == 16 * [3]
    assert nx.diameter(G) == 4
    G = nx.octahedral_graph()
    assert sorted(G) == list(range(6))
    assert G.number_of_edges() == 12
    assert [d for n, d in G.degree()] == 6 * [4]
    assert nx.diameter(G) == 2
    assert nx.radius(G) == 2
    G = nx.pappus_graph()
    assert sorted(G) == list(range(18))
    assert G.number_of_edges() == 27
    assert [d for n, d in G.degree()] == 18 * [3]
    assert nx.diameter(G) == 4
    G = nx.petersen_graph()
    assert sorted(G) == list(range(10))
    assert G.number_of_edges() == 15
    assert [d for n, d in G.degree()] == 10 * [3]
    assert nx.diameter(G) == 2
    assert nx.radius(G) == 2
    G = nx.sedgewick_maze_graph()
    assert sorted(G) == list(range(8))
    assert G.number_of_edges() == 10
    assert sorted((d for n, d in G.degree())) == [1, 2, 2, 2, 3, 3, 3, 4]
    G = nx.tetrahedral_graph()
    assert sorted(G) == list(range(4))
    assert G.number_of_edges() == 6
    assert [d for n, d in G.degree()] == [3, 3, 3, 3]
    assert nx.diameter(G) == 1
    assert nx.radius(G) == 1
    G = nx.truncated_cube_graph()
    assert sorted(G) == list(range(24))
    assert G.number_of_edges() == 36
    assert [d for n, d in G.degree()] == 24 * [3]
    G = nx.truncated_tetrahedron_graph()
    assert sorted(G) == list(range(12))
    assert G.number_of_edges() == 18
    assert [d for n, d in G.degree()] == 12 * [3]
    G = nx.tutte_graph()
    assert sorted(G) == list(range(46))
    assert G.number_of_edges() == 69
    assert [d for n, d in G.degree()] == 46 * [3]
    pytest.raises(nx.NetworkXError, nx.tutte_graph, create_using=nx.DiGraph)
    MG = nx.tutte_graph(create_using=nx.MultiGraph)
    assert sorted(MG.edges()) == sorted(G.edges())