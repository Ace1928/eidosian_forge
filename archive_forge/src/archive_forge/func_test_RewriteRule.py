from __future__ import annotations
from dask.rewrite import VAR, RewriteRule, RuleSet, Traverser, args, head
from dask.utils_test import add, inc
def test_RewriteRule():
    assert rule1.vars == ('a',)
    assert rule1._varlist == ['a']
    assert rule2.vars == ('a',)
    assert rule2._varlist == ['a', 'a']
    assert rule3.vars == ('a',)
    assert rule3._varlist == ['a', 'a']
    assert rule4.vars == ('a', 'b')
    assert rule4._varlist == ['b', 'a']
    assert rule5.vars == ('a', 'b', 'c')
    assert rule5._varlist == ['c', 'b', 'a']