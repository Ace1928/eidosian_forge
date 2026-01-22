import itertools
import numpy as np
import pytest
import cirq
def test_by_index():
    eq = cirq.testing.EqualsTester()
    eq.add_equality_group(cirq.X, *[cirq.Pauli.by_index(i) for i in (-3, 0, 3, 6)])
    eq.add_equality_group(cirq.Y, *[cirq.Pauli.by_index(i) for i in (-2, 1, 4, 7)])
    eq.add_equality_group(cirq.Z, *[cirq.Pauli.by_index(i) for i in (-1, 2, 5, 8)])