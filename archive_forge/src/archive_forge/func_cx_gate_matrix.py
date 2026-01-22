from __future__ import annotations
from string import ascii_uppercase, ascii_lowercase
import numpy as np
import qiskit.circuit.library.standard_gates as gates
from qiskit.exceptions import QiskitError
def cx_gate_matrix() -> np.ndarray:
    """Get the matrix for a controlled-NOT gate."""
    return _CX_MATRIX