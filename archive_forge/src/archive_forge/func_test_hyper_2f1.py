import math
import pytest
from mpmath import *
def test_hyper_2f1():
    mp.dps = 15
    v = 1.0652207633823292
    assert hyper([(1, 2), (3, 4)], [2], 0.3).ae(v)
    assert hyper([(1, 2), 0.75], [2], 0.3).ae(v)
    assert hyper([0.5, 0.75], [2.0], 0.3).ae(v)
    assert hyper([0.5, 0.75], [2.0], 0.3 + 0j).ae(v)
    assert hyper([0.5 + 0j, (3, 4)], [2.0], 0.3 + 0j).ae(v)
    assert hyper([0.5 + 0j, (3, 4)], [2.0], 0.3).ae(v)
    assert hyper([0.5, (3, 4)], [2.0 + 0j], 0.3).ae(v)
    assert hyper([0.5 + 0j, 0.75 + 0j], [2.0 + 0j], 0.3 + 0j).ae(v)
    v = 1.0923468109622323 + 0.1810485916947936j
    assert hyper([(1, 2), 0.75 + j], [2], 0.5).ae(v)
    assert hyper([0.5, 0.75 + j], [2.0], 0.5).ae(v)
    assert hyper([0.5, 0.75 + j], [2.0], 0.5 + 0j).ae(v)
    assert hyper([0.5, 0.75 + j], [2.0 + 0j], 0.5 + 0j).ae(v)
    v = 0.9625 - 0.125j
    assert hyper([(3, 2), -1], [4], 0.1 + j / 3).ae(v)
    assert hyper([1.5, -1.0], [4], 0.1 + j / 3).ae(v)
    assert hyper([1.5, -1.0], [4 + 0j], 0.1 + j / 3).ae(v)
    assert hyper([1.5 + 0j, -1.0 + 0j], [4 + 0j], 0.1 + j / 3).ae(v)
    v = 1.0211106950169344 - 0.5040225261346686j
    assert hyper([(2, 10), (3, 10)], [(4, 10)], 1.5).ae(v)
    assert hyper([0.2, (3, 10)], [0.4 + 0j], 1.5).ae(v)
    assert hyper([0.2, (3, 10)], [0.4 + 0j], 1.5 + 0j).ae(v)
    v = 0.7692250136286585 + 0.32640579593235886j
    assert hyper([(2, 10), (3, 10)], [(4, 10)], 4 + 2j).ae(v)
    assert hyper([0.2, (3, 10)], [0.4 + 0j], 4 + 2j).ae(v)
    assert hyper([0.2, (3, 10)], [(4, 10)], 4 + 2j).ae(v)