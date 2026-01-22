from __future__ import annotations
from typing import List, Optional
import numpy as np
from qiskit.circuit import QuantumRegister, AncillaRegister, QuantumCircuit
from qiskit.circuit.exceptions import CircuitError
from .functional_pauli_rotations import FunctionalPauliRotations
from .polynomial_pauli_rotations import PolynomialPauliRotations
from .integer_comparator import IntegerComparator
@property
def mapped_coeffs(self) -> List[List[float]]:
    """The coefficients mapped to the internal representation, since we only compare
        x>=breakpoint.

        Returns:
            The mapped coefficients.
        """
    mapped_coeffs = []
    mapped_coeffs.append(self._hom_coeffs[0])
    for i in range(1, len(self._hom_coeffs)):
        mapped_coeffs.append([])
        for j in range(0, self._degree + 1):
            mapped_coeffs[i].append(self._hom_coeffs[i][j] - self._hom_coeffs[i - 1][j])
    return mapped_coeffs