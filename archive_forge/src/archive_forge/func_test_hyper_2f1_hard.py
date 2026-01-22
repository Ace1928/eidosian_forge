import math
import pytest
from mpmath import *
def test_hyper_2f1_hard():
    mp.dps = 15
    assert hyp2f1(2, -1, -1, 3).ae(7)
    assert hyp2f1(2, -1, -1, 3, eliminate_all=True).ae(0.25)
    assert hyp2f1(2, -2, -2, 3).ae(34)
    assert hyp2f1(2, -2, -2, 3, eliminate_all=True).ae(0.25)
    assert hyp2f1(2, -2, -3, 3) == 14
    assert hyp2f1(2, -3, -2, 3) == inf
    assert hyp2f1(2, -1.5, -1.5, 3) == 0.25
    assert hyp2f1(1, 2, 3, 0) == 1
    assert hyp2f1(0, 1, 0, 0) == 1
    assert hyp2f1(0, 0, 0, 0) == 1
    assert isnan(hyp2f1(1, 1, 0, 0))
    assert hyp2f1(2, -1, -5, 0.25 + 0.25j).ae(1.1 + 0.1j)
    assert hyp2f1(2, -5, -5, 0.25 + 0.25j, eliminate=False).ae(163.0 / 128 + 125.0 / 128 * j)
    assert hyp2f1(0.7235, -1, -5, 0.3).ae(1.04341)
    assert hyp2f1(0.7235, -5, -5, 0.3, eliminate=False).ae(1.2939225017815903)
    assert hyp2f1(-1, -2, 4, 1) == 1.5
    assert hyp2f1(1, 2, -3, 1) == inf
    assert hyp2f1(-2, -2, 1, 1) == 6
    assert hyp2f1(1, -2, -4, 1).ae(5.0 / 3)
    assert hyp2f1(0, -6, -4, 1) == 1
    assert hyp2f1(0, -3, -4, 1) == 1
    assert hyp2f1(0, 0, 0, 1) == 1
    assert hyp2f1(1, 0, 0, 1, eliminate=False) == 1
    assert hyp2f1(1, 1, 0, 1) == inf
    assert hyp2f1(1, -6, -4, 1) == inf
    assert hyp2f1(-7.2, -0.5, -4.5, 1) == 0
    assert hyp2f1(-7.2, -1, -2, 1).ae(-2.6)
    assert hyp2f1(1, -0.5, -4.5, 1) == inf
    assert hyp2f1(1, 0.5, -4.5, 1) == -inf
    z = exp(j * pi / 3)
    w = (nthroot(2, 3) + 1) * exp(j * pi / 12) / nthroot(3, 4) ** 3
    assert hyp2f1('1/2', '1/6', '1/3', z).ae(w)
    assert hyp2f1('1/2', '1/6', '1/3', z.conjugate()).ae(w.conjugate())
    assert hyp2f1(0.25, (1, 3), 2, '0.999').ae(1.0682644949603064)
    assert hyp2f1(0.25, (1, 3), 2, '1.001').ae(1.0686729925483032 - 1.446586793975874e-05j)
    assert hyp2f1(0.25, (1, 3), 2, -1).ae(0.9665658449252436)
    assert hyp2f1(0.25, (1, 3), 2, j).ae(0.9904176624898208 + 0.037771356041807355j)
    assert hyp2f1(2, 3, 5, '0.99').ae(27.69934790432269)
    assert hyp2f1((3, 2), -0.5, 3, '0.99').ae(0.6840303684391166)
    assert hyp2f1(2, 3, 5, 1j).ae(0.3729066714597439 + 0.5921000490274828j)
    assert fsum([hyp2f1((7, 10), (2, 3), (-1, 2), 0.95 * exp(j * k)) for k in range(1, 15)]).ae(52.851400204289455 + 6.244285013912953j)
    assert fsum([hyp2f1((7, 10), (2, 3), (-1, 2), 1.05 * exp(j * k)) for k in range(1, 15)]).ae(54.506013786220656 - 3.0001188134132173j)
    assert fsum([hyp2f1((7, 10), (2, 3), (-1, 2), exp(j * k)) for k in range(1, 15)]).ae(55.79207793595531 + 1.7319864857785003j)
    assert hyp2f1(2, 2.5, -3.25, 0.999).ae(2.183739328012171e+26)
    assert hyp2f1(1, 1, 2, 1.01).ae(4.559574441572368 - 3.1104877758314786j)
    assert hyp2f1(1, 1, 2, 1.01 + 0.1j).ae(2.414942748055278 + 1.414822479683694j)
    assert hyp2f1(1, 1, 2, 3 + 4j).ae(0.14576709331407298 + 0.4837918541798036j)
    assert hyp2f1(1, 1, 2, 4).ae(-0.27465307216702745 - 0.7853981633974483j)
    assert hyp2f1(1, 1, 2, -4).ae(0.40235947810852507)
    assert hyp2f1(112, (51, 10), (-9, 10), -0.99999).ae(-1.6241361047970863e-24, abs_eps=0, rel_eps=eps * 16)