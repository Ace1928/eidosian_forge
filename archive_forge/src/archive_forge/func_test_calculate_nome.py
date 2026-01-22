import mpmath
import random
import pytest
from mpmath import *
def test_calculate_nome():
    mp.dps = 100
    q = calculate_nome(zero)
    assert q == zero
    mp.dps = 25
    math1 = [(mpf(1) / 10, mpf('0.006584651553858370274473060')), (mpf(2) / 10, mpf('0.01394285727531826872146409')), (mpf(3) / 10, mpf('0.02227743615715350822901627')), (mpf(4) / 10, mpf('0.03188334731336317755064299')), (mpf(5) / 10, mpf('0.04321391826377224977441774')), (mpf(6) / 10, mpf('0.05702025781460967637754953')), (mpf(7) / 10, mpf('0.07468994353717944761143751')), (mpf(8) / 10, mpf('0.09927369733882489703607378')), (mpf(9) / 10, mpf('0.1401731269542615524091055')), (mpf(9) / 10, mpf('0.1401731269542615524091055'))]
    for i in math1:
        m = i[0]
        q = calculate_nome(sqrt(m))
        assert q.ae(i[1])
    mp.dps = 15