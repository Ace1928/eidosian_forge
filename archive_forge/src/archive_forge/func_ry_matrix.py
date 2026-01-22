import numpy as np
from qiskit.circuit.library.standard_gates import RXGate, RZGate, RYGate
def ry_matrix(phi: float) -> np.ndarray:
    """
    Computes an RY rotation by the angle of ``phi``.

    Args:
        phi: rotation angle.

    Returns:
        an RY rotation matrix.
    """
    return RYGate(phi).to_matrix()