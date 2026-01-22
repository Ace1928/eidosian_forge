from __future__ import annotations
from dask.rewrite import VAR, RewriteRule, RuleSet, Traverser, args, head
from dask.utils_test import add, inc
def test_traverser():
    term = (add, (inc, 1), (double, (inc, 1), 2))
    t = Traverser(term)
    t2 = t.copy()
    assert t.current == add
    t.next()
    assert t.current == inc
    assert t2.current == add
    t.skip()
    assert t.current == double
    t.next()
    assert t.current == inc
    assert list(t2) == [add, inc, 1, double, inc, 1, 2]