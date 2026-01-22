from __future__ import division
import pytest
from mpmath import *
def test_exp_pade():
    for i in range(3):
        dps = 15
        extra = 15
        mp.dps = dps + extra
        dm = 0
        N = 3
        dg = range(1, N + 1)
        a = diag(dg)
        expa = diag([exp(x) for x in dg])
        while abs(dm) < 0.01:
            m = randmatrix(N)
            dm = det(m)
        m = m / dm
        a1 = m ** (-1) * a * m
        e2 = m ** (-1) * expa * m
        mp.dps = dps
        e1 = expm(a1, method='pade')
        mp.dps = dps + extra
        d = e2 - e1
        mp.dps = dps
        assert norm(d, inf).ae(0)
    mp.dps = 15