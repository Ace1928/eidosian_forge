import numpy as np
from qiskit.circuit.library.standard_gates import RXGate, RZGate, RYGate
def rx_matrix(phi: float) -> np.ndarray:
    """
    Computes an RX rotation by the angle of ``phi``.

    Args:
        phi: rotation angle.

    Returns:
        an RX rotation matrix.
    """
    return RXGate(phi).to_matrix()