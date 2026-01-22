from __future__ import annotations
from typing import Callable
import numpy as np
from numpy.polynomial.chebyshev import Chebyshev
from qiskit.circuit import QuantumRegister, AncillaRegister
from qiskit.circuit.library.blueprintcircuit import BlueprintCircuit
from qiskit.circuit.exceptions import CircuitError
from .piecewise_polynomial_pauli_rotations import PiecewisePolynomialPauliRotations
@f_x.setter
def f_x(self, f_x: float | Callable[[int], float] | None) -> None:
    """Set the function to be approximated.

        Note that this may change the underlying quantum register, if the number of state qubits
        changes.

        Args:
            f_x: The new function to be approximated.
        """
    if self._f_x is None or f_x != self._f_x:
        self._invalidate()
        self._f_x = f_x
        self._reset_registers(self.num_state_qubits)