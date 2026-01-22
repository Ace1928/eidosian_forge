import pytest
import rpy2.robjects as robjects
import array
def test_items():
    env = robjects.Environment()
    env['a'] = 123
    env['b'] = 234
    items = list(env.items())
    assert len(items) == 2
    items.sort(key=lambda x: x[0])
    for it_a, it_b in zip(items, (('a', 123), ('b', 234))):
        assert it_a[0] == it_b[0]
        assert it_a[1][0] == it_b[1]