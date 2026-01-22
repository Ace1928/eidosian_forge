import itertools
import os
import time
from collections import defaultdict
from random import randint, random, sample, randrange
from typing import Iterator, Optional, Tuple, TYPE_CHECKING
import numpy as np
import pytest
import sympy
import cirq
from cirq import circuits
from cirq import ops
from cirq.testing.devices import ValidatingTestDevice
@pytest.mark.parametrize('rs, n_qubits', (([0.1, 0.2], 1), ([0.1, 0.2], 2), ([0.8, 0.9], 1), ([0.8, 0.9], 2), ([0.1, 0.2, 0.3], 1), ([0.1, 0.2, 0.3], 2), ([0.1, 0.2, 0.3], 3)))
def test_circuit_superoperator_depolarizing_channel_compositions(rs, n_qubits):
    """Tests Circuit._superoperator_() on compositions of depolarizing channels."""

    def pauli_error_probability(r: float, n_qubits: int) -> float:
        """Computes Pauli error probability for given depolarization parameter.

        Pauli error is what cirq.depolarize takes as argument. Depolarization parameter
        makes it simple to compute the serial composition of depolarizing channels. It
        is multiplicative under channel composition.
        """
        d2 = 4 ** n_qubits
        return (1 - r) * (d2 - 1) / d2

    def depolarize(r: float, n_qubits: int) -> cirq.DepolarizingChannel:
        """Returns depolarization channel with given depolarization parameter."""
        return cirq.depolarize(pauli_error_probability(r, n_qubits=n_qubits), n_qubits=n_qubits)
    qubits = cirq.LineQubit.range(n_qubits)
    circuit1 = cirq.Circuit((depolarize(r, n_qubits).on(*qubits) for r in rs))
    circuit2 = cirq.Circuit(depolarize(np.prod(rs), n_qubits).on(*qubits))
    assert circuit1._has_superoperator_()
    assert circuit2._has_superoperator_()
    cm1 = circuit1._superoperator_()
    cm2 = circuit2._superoperator_()
    assert np.allclose(cm1, cm2)