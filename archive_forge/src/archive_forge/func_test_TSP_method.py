import random
import pytest
import networkx as nx
import networkx.algorithms.approximation as nx_app
def test_TSP_method():
    G = nx.cycle_graph(9)
    G[4][5]['weight'] = 10

    def my_tsp_method(G, weight):
        return nx_app.simulated_annealing_tsp(G, 'greedy', weight, source=4, seed=1)
    path = nx_app.traveling_salesman_problem(G, method=my_tsp_method, cycle=False)
    print(path)
    assert path == [4, 3, 2, 1, 0, 8, 7, 6, 5]