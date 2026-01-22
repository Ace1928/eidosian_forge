from __future__ import annotations
from typing import Callable
import numpy as np
from numpy.polynomial.chebyshev import Chebyshev
from qiskit.circuit import QuantumRegister, AncillaRegister
from qiskit.circuit.library.blueprintcircuit import BlueprintCircuit
from qiskit.circuit.exceptions import CircuitError
from .piecewise_polynomial_pauli_rotations import PiecewisePolynomialPauliRotations
@polynomials.setter
def polynomials(self, polynomials: list[list[float]] | None) -> None:
    """Set the polynomials for the piecewise approximation.

        Note that this may change the underlying quantum register, if the number of state qubits
        changes.

        Args:
            polynomials: The new breakpoints for the piecewise approximation.
        """
    if self._polynomials is None or polynomials != self._polynomials:
        self._invalidate()
        self._polynomials = polynomials
        self._reset_registers(self.num_state_qubits)