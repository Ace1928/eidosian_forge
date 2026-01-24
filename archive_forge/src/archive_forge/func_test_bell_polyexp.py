from mpmath import *
from mpmath.libmp import round_up, from_float, mpf_zeta_int
def test_bell_polyexp():
    mp.dps = 15
    assert (polyexp(0, 1e-10) * 10 ** 10).ae(1.00000000005)
    assert (polyexp(1, 1e-10) * 10 ** 10).ae(1.0000000001)
    assert polyexp(5, 3j).ae(-607.7044517476177 + 519.9627864820014j)
    assert polyexp(-1, 3.5).ae(12.095375361755435)
    assert bell(0, 0) == 1
    assert bell(0, 1) == 1
    assert bell(0, 2) == 1
    assert bell(0, inf) == 1
    assert bell(0, -inf) == 1
    assert isnan(bell(0, nan))
    assert bell(1, 4) == 4
    assert bell(1, 0) == 0
    assert bell(1, inf) == inf
    assert bell(1, -inf) == -inf
    assert isnan(bell(1, nan))
    assert bell(2, -1) == 0
    assert bell(2, 0) == 0
    assert bell(10) == 115975
    assert bell(10, 1) == 115975
    assert bell(10, -8) == 11054008
    assert bell(5, -50) == -253087550
    assert bell(50, -50).ae('3.4746902914629720259e74')
    mp.dps = 80
    assert bell(50, -50) == 347469029146297202586097646631767227177164818163463279814268368579055777450
    assert bell(40, 50) == 5575520134721105844739265207408344706846955281965031698189286176321717550
    assert bell(74) == 5006908024247925379707076470957722220463116781409659160159536981161298714301202
    mp.dps = 15
    assert bell(10, 20j) == 7504528595600 + 15649605360020j
    assert bell(0.5, 0).ae(sinc(pi * 0.5))