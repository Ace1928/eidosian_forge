from itertools import combinations
import pytest
import networkx as nx
def test_is_minimal_d_separator(large_collider_graph, chain_and_fork_graph, no_separating_set_graph, large_no_separating_set_graph, collider_trek_graph):
    assert not nx.is_d_separator(large_collider_graph, {'B'}, {'E'}, set())
    Zmin = nx.find_minimal_d_separator(large_collider_graph, 'B', 'E')
    assert nx.is_d_separator(large_collider_graph, 'B', 'E', Zmin)
    assert nx.is_minimal_d_separator(large_collider_graph, 'B', 'E', Zmin)
    assert nx.is_minimal_d_separator(large_collider_graph, {'A', 'B'}, {'G', 'E'}, Zmin)
    assert Zmin == {'D'}
    assert not nx.is_d_separator(chain_and_fork_graph, {'A'}, {'C'}, set())
    Zmin = nx.find_minimal_d_separator(chain_and_fork_graph, 'A', 'C')
    assert nx.is_minimal_d_separator(chain_and_fork_graph, 'A', 'C', Zmin)
    assert Zmin == {'B'}
    Znotmin = Zmin.union({'D'})
    assert not nx.is_minimal_d_separator(chain_and_fork_graph, 'A', 'C', Znotmin)
    assert not nx.is_d_separator(no_separating_set_graph, {'A'}, {'B'}, set())
    assert nx.find_minimal_d_separator(no_separating_set_graph, 'A', 'B') is None
    assert not nx.is_d_separator(large_no_separating_set_graph, {'A'}, {'B'}, {'C'})
    assert nx.find_minimal_d_separator(large_no_separating_set_graph, 'A', 'B') is None
    assert nx.find_minimal_d_separator(collider_trek_graph, 'A', 'D', included='B') == {'B', 'C'}
    assert nx.find_minimal_d_separator(collider_trek_graph, 'A', 'D', included='B', restricted='B') is None