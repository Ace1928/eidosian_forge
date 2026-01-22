import random
import pytest
import networkx as nx
import networkx.algorithms.approximation as nx_app
def test_two_nodes(self):
    G = nx.Graph()
    G.add_weighted_edges_from({(1, 2, 1)})
    cycle = self.tsp(G, 'greedy', source=1, seed=42)
    cost = sum((G[n][nbr]['weight'] for n, nbr in pairwise(cycle)))
    validate_solution(cycle, cost, [1, 2, 1], 2)
    cycle = self.tsp(G, [1, 2, 1], source=1, seed=42)
    cost = sum((G[n][nbr]['weight'] for n, nbr in pairwise(cycle)))
    validate_solution(cycle, cost, [1, 2, 1], 2)