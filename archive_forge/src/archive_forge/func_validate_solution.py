import random
import pytest
import networkx as nx
import networkx.algorithms.approximation as nx_app
def validate_solution(soln, cost, exp_soln, exp_cost):
    assert soln == exp_soln
    assert cost == exp_cost