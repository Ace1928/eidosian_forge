import math
import pytest
from mpmath import *
def test_hyper_1f1():
    mp.dps = 15
    v = 1.2917526488617657
    assert hyper([(1, 2)], [(3, 2)], 0.7).ae(v)
    assert hyper([(1, 2)], [(3, 2)], 0.7 + 0j).ae(v)
    assert hyper([0.5], [(3, 2)], 0.7).ae(v)
    assert hyper([0.5], [1.5], 0.7).ae(v)
    assert hyper([0.5], [(3, 2)], 0.7 + 0j).ae(v)
    assert hyper([0.5], [1.5], 0.7 + 0j).ae(v)
    assert hyper([(1, 2)], [1.5 + 0j], 0.7).ae(v)
    assert hyper([0.5 + 0j], [1.5], 0.7).ae(v)
    assert hyper([0.5 + 0j], [1.5 + 0j], 0.7 + 0j).ae(v)
    assert hyp1f1(0.5, 1.5, 0.7).ae(v)
    assert hyp1f1((1, 2), 1.5, 0.7).ae(v)
    assert hyp1f1(2, 3, 10000000000.0).ae('2.1555012157015796988e+4342944809')
    assert (hyp1f1(2, 3, 10000000000j) * 10 ** 10).ae(-0.9750120502003975 - 1.7462392454512132j)
    assert hyp1f1(-2, 1, 10000).ae(49980001)
    assert hyp1f1(1j, fraction(1, 3), 0.415 - 69.739j).ae(25.857588206024346 + 15.738060264515292j)