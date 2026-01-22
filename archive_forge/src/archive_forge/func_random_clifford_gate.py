import itertools
import numpy as np
import pytest
import sympy
import cirq
import cirq.testing
def random_clifford_gate():
    matrix = np.eye(2)
    for _ in range(10):
        matrix = matrix @ cirq.unitary(np.random.choice((cirq.H, cirq.S)))
    matrix *= np.exp(1j * np.random.uniform(0, 2 * np.pi))
    return cirq.MatrixGate(matrix)