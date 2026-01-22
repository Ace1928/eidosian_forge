import array
import pytest
import rpy2.robjects as robjects
def test_slots():
    x = robjects.r('list(a=1,b=2,c=3)')
    s = x.slots
    assert len(s) == 1
    assert tuple(s.keys()) == ('names',)
    assert tuple(s['names']) == ('a', 'b', 'c')
    s['names'] = 0
    assert len(s) == 1
    assert tuple(s.keys()) == ('names',)
    assert tuple(s['names']) == (0,)