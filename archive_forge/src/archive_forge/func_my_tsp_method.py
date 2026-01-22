import random
import pytest
import networkx as nx
import networkx.algorithms.approximation as nx_app
def my_tsp_method(G, weight):
    return nx_app.simulated_annealing_tsp(G, 'greedy', weight, source=4, seed=1)