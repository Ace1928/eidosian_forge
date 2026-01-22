import random
import pytest
import networkx as nx
import networkx.algorithms.approximation as nx_app
def test_TSP_weighted():
    G = nx.cycle_graph(9)
    G[0][1]['weight'] = 2
    G[1][2]['weight'] = 2
    G[2][3]['weight'] = 2
    G[3][4]['weight'] = 4
    G[4][5]['weight'] = 5
    G[5][6]['weight'] = 4
    G[6][7]['weight'] = 2
    G[7][8]['weight'] = 2
    G[8][0]['weight'] = 2
    tsp = nx_app.traveling_salesman_problem
    expected_paths = ([3, 2, 1, 0, 8, 7, 6], [6, 7, 8, 0, 1, 2, 3])
    expected_cycles = ([3, 2, 1, 0, 8, 7, 6, 7, 8, 0, 1, 2, 3], [6, 7, 8, 0, 1, 2, 3, 2, 1, 0, 8, 7, 6])
    expected_tourpaths = ([5, 6, 7, 8, 0, 1, 2, 3, 4], [4, 3, 2, 1, 0, 8, 7, 6, 5])
    cycle = tsp(G, nodes=[3, 6], weight='weight')
    assert cycle in expected_cycles
    path = tsp(G, nodes=[3, 6], weight='weight', cycle=False)
    assert path in expected_paths
    tourpath = tsp(G, weight='weight', cycle=False)
    assert tourpath in expected_tourpaths
    methods = [nx_app.christofides, nx_app.greedy_tsp, lambda G, wt: nx_app.simulated_annealing_tsp(G, 'greedy', weight=wt), lambda G, wt: nx_app.threshold_accepting_tsp(G, 'greedy', weight=wt)]
    for method in methods:
        cycle = tsp(G, nodes=[3, 6], weight='weight', method=method)
        assert cycle in expected_cycles
        path = tsp(G, nodes=[3, 6], weight='weight', method=method, cycle=False)
        assert path in expected_paths
        tourpath = tsp(G, weight='weight', method=method, cycle=False)
        assert tourpath in expected_tourpaths