from __future__ import annotations
import numpy as np
from qiskit.circuit import QuantumRegister, AncillaRegister, QuantumCircuit
from qiskit.circuit.exceptions import CircuitError
from .functional_pauli_rotations import FunctionalPauliRotations
from .linear_pauli_rotations import LinearPauliRotations
from .integer_comparator import IntegerComparator
@property
def mapped_slopes(self) -> np.ndarray:
    """The slopes mapped to the internal representation.

        Returns:
            The mapped slopes.
        """
    mapped_slopes = np.zeros_like(self.slopes)
    for i, slope in enumerate(self.slopes):
        mapped_slopes[i] = slope - sum(mapped_slopes[:i])
    return mapped_slopes