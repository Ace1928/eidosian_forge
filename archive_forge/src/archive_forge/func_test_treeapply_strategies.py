from sympy.strategies.tree import treeapply, greedy, allresults, brute
from functools import partial, reduce
def test_treeapply_strategies():
    from sympy.strategies import chain, minimize
    join = {list: chain, tuple: minimize}
    assert treeapply(inc, join) == inc
    assert treeapply((inc, dec), join)(5) == minimize(inc, dec)(5)
    assert treeapply([inc, dec], join)(5) == chain(inc, dec)(5)
    tree = (inc, [dec, double])
    assert treeapply(tree, join)(5) == 6
    assert treeapply(tree, join)(1) == 0
    maximize = partial(minimize, objective=lambda x: -x)
    join = {list: chain, tuple: maximize}
    fn = treeapply(tree, join)
    assert fn(4) == 6
    assert fn(1) == 2