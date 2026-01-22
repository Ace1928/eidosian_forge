import collections.abc
import pathlib
import numpy as np
import pytest
import sympy
import cirq
import cirq.testing
def test_mixture():
    a = cirq.NamedQubit('a')
    op = cirq.bit_flip(0.5).on(a)
    assert_mixtures_equal(cirq.mixture(op), cirq.mixture(op.gate))
    assert cirq.has_mixture(op)
    assert cirq.has_mixture(cirq.X(a))
    m = cirq.mixture(cirq.X(a))
    assert len(m) == 1
    assert m[0][0] == 1
    np.testing.assert_allclose(m[0][1], cirq.unitary(cirq.X))