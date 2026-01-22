import pytest
import networkx as nx
def test_discard_issue(self):
    g = nx.DiGraph()
    g.add_edges_from([('b0', 'b1'), ('b1', 'b2'), ('b2', 'b3'), ('b3', 'b1'), ('b1', 'b5'), ('b5', 'b6'), ('b5', 'b8'), ('b6', 'b7'), ('b8', 'b7'), ('b7', 'b3'), ('b3', 'b4')])
    df = nx.dominance_frontiers(g, 'b0')
    assert df == {'b4': set(), 'b5': {'b3'}, 'b6': {'b7'}, 'b7': {'b3'}, 'b0': set(), 'b1': {'b1'}, 'b2': {'b3'}, 'b3': {'b1'}, 'b8': {'b7'}}