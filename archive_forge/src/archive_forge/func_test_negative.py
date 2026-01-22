import random
import time
import networkx as nx
from networkx.algorithms.isomorphism.tree_isomorphism import (
from networkx.classes.function import is_directed
def test_negative(maxk=11):
    print('negative test')
    for k in range(4, maxk + 1):
        test_trees = list(nx.nonisomorphic_trees(k))
        start_time = time.time()
        trial = 0
        for i in range(len(test_trees) - 1):
            for j in range(i + 1, len(test_trees)):
                trial += 1
                assert tree_isomorphism(test_trees[i], test_trees[j]) == []
        print(k, trial, time.time() - start_time)