import numpy as np
import pytest
import sympy
import cirq
def test_zz_phase_by():
    assert cirq.phase_by(cirq.ZZ, 0.25, 0) == cirq.phase_by(cirq.ZZ, 0.25, 1) == cirq.ZZ
    assert cirq.phase_by(cirq.ZZ ** 0.5, 0.25, 0) == cirq.ZZ ** 0.5
    assert cirq.phase_by(cirq.ZZ ** (-0.5), 0.25, 1) == cirq.ZZ ** (-0.5)