import numpy as np
import pytest
import sympy
import cirq
def test_zz_pow():
    assert cirq.ZZ ** 0.5 != cirq.ZZ ** (-0.5)
    assert cirq.ZZ ** (-1) == cirq.ZZ
    assert (cirq.ZZ ** (-1)) ** 0.5 == cirq.ZZ ** (-0.5)