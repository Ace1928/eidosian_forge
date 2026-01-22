import pytest
import networkx as nx
from networkx.algorithms.isomorphism.isomorph import graph_could_be_isomorphic
@pytest.mark.parametrize('fn', (nx.bull_graph, nx.chvatal_graph, nx.cubical_graph, nx.diamond_graph, nx.house_graph, nx.house_x_graph, nx.icosahedral_graph, nx.krackhardt_kite_graph, nx.octahedral_graph, nx.petersen_graph, nx.truncated_cube_graph, nx.tutte_graph))
@pytest.mark.parametrize('create_using', (nx.DiGraph, nx.MultiDiGraph, nx.DiGraph([(0, 1)])))
def tests_raises_with_directed_create_using(fn, create_using):
    with pytest.raises(nx.NetworkXError, match='Directed Graph not supported'):
        fn(create_using=create_using)