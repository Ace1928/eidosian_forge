import bz2
import importlib.resources
import os
import pickle
import pytest
import networkx as nx
def test_google_or_tools_example2():
    """
    https://developers.google.com/optimization/flow/mincostflow
    """
    G = nx.DiGraph()
    start_nodes = [0, 0, 1, 1, 1, 2, 2, 3, 4, 3]
    end_nodes = [1, 2, 2, 3, 4, 3, 4, 4, 2, 5]
    capacities = [15, 8, 20, 4, 10, 15, 4, 20, 5, 10]
    unit_costs = [4, 4, 2, 2, 6, 1, 3, 2, 3, 4]
    supplies = [23, 0, 0, -5, -15, -3]
    answer = 183
    for i in range(len(supplies)):
        G.add_node(i, demand=-1 * supplies[i])
    for i in range(len(start_nodes)):
        G.add_edge(start_nodes[i], end_nodes[i], weight=unit_costs[i], capacity=capacities[i])
    flowCost, flowDict = nx.network_simplex(G)
    assert flowCost == answer
    assert flowCost == get_flowcost_from_flowdict(G, flowDict)