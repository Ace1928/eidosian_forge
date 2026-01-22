import pytest
import rpy2.robjects as robjects
import array
def test_values():
    env = robjects.Environment()
    env['a'] = 123
    env['b'] = 234
    values = list(env.values())
    assert len(values) == 2
    values.sort(key=lambda x: x[0])
    for it_a, it_b in zip(values, (123, 234)):
        assert len(it_a) == 1
        assert it_a[0] == it_b