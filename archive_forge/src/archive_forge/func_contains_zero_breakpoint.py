from __future__ import annotations
import numpy as np
from qiskit.circuit import QuantumRegister, AncillaRegister, QuantumCircuit
from qiskit.circuit.exceptions import CircuitError
from .functional_pauli_rotations import FunctionalPauliRotations
from .linear_pauli_rotations import LinearPauliRotations
from .integer_comparator import IntegerComparator
@property
def contains_zero_breakpoint(self) -> bool | np.bool_:
    """Whether 0 is the first breakpoint.

        Returns:
            True, if 0 is the first breakpoint, otherwise False.
        """
    return np.isclose(0, self.breakpoints[0])