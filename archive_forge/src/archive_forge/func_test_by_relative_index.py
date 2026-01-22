import itertools
import numpy as np
import pytest
import cirq
def test_by_relative_index():
    assert cirq.Pauli.by_relative_index(cirq.X, -1) == cirq.Z
    assert cirq.Pauli.by_relative_index(cirq.X, 0) == cirq.X
    assert cirq.Pauli.by_relative_index(cirq.X, 1) == cirq.Y
    assert cirq.Pauli.by_relative_index(cirq.X, 2) == cirq.Z
    assert cirq.Pauli.by_relative_index(cirq.X, 3) == cirq.X
    assert cirq.Pauli.by_relative_index(cirq.Y, -1) == cirq.X
    assert cirq.Pauli.by_relative_index(cirq.Y, 0) == cirq.Y
    assert cirq.Pauli.by_relative_index(cirq.Y, 1) == cirq.Z
    assert cirq.Pauli.by_relative_index(cirq.Y, 2) == cirq.X
    assert cirq.Pauli.by_relative_index(cirq.Y, 3) == cirq.Y
    assert cirq.Pauli.by_relative_index(cirq.Z, -1) == cirq.Y
    assert cirq.Pauli.by_relative_index(cirq.Z, 0) == cirq.Z
    assert cirq.Pauli.by_relative_index(cirq.Z, 1) == cirq.X
    assert cirq.Pauli.by_relative_index(cirq.Z, 2) == cirq.Y
    assert cirq.Pauli.by_relative_index(cirq.Z, 3) == cirq.Z