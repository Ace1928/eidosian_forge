from sympy.strategies.tree import treeapply, greedy, allresults, brute
from functools import partial, reduce
def test_treeapply_leaf():
    assert treeapply(3, {}, leaf=lambda x: x ** 2) == 9
    tree = ([3, 3], [4, 1], 2)
    treep1 = ([4, 4], [5, 2], 3)
    assert treeapply(tree, {list: min, tuple: max}, leaf=lambda x: x + 1) == treeapply(treep1, {list: min, tuple: max})