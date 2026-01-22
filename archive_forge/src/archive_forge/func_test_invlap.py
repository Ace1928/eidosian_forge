import pytest
from mpmath import *
def test_invlap():
    mp.dps = 15
    t = 0.01
    fp = lambda p: 1 / (p + 1) ** 2
    ft = lambda t: t * exp(-t)
    ftt = ft(t)
    assert invertlaplace(fp, t, method='talbot').ae(ftt)
    assert invertlaplace(fp, t, method='stehfest').ae(ftt)
    assert invertlaplace(fp, t, method='dehoog').ae(ftt)
    assert invertlaplace(fp, t, method='cohen').ae(ftt)
    t = 1.0
    ftt = ft(t)
    assert invertlaplace(fp, t, method='talbot').ae(ftt)
    assert invertlaplace(fp, t, method='stehfest').ae(ftt)
    assert invertlaplace(fp, t, method='dehoog').ae(ftt)
    assert invertlaplace(fp, t, method='cohen').ae(ftt)
    t = 0.01
    fp = lambda p: log(p) / p
    ft = lambda t: -euler - log(t)
    ftt = ft(t)
    assert invertlaplace(fp, t, method='talbot').ae(ftt)
    assert invertlaplace(fp, t, method='stehfest').ae(ftt)
    assert invertlaplace(fp, t, method='dehoog').ae(ftt)
    assert invertlaplace(fp, t, method='cohen').ae(ftt)
    t = 1.0
    ftt = ft(t)
    assert invertlaplace(fp, t, method='talbot').ae(ftt)
    assert invertlaplace(fp, t, method='stehfest').ae(ftt)
    assert invertlaplace(fp, t, method='dehoog').ae(ftt)
    assert invertlaplace(fp, t, method='cohen').ae(ftt)