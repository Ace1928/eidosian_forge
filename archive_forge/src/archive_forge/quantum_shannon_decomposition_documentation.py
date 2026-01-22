from typing import List, Callable, TYPE_CHECKING
from scipy.linalg import cossin
import numpy as np
from cirq import ops
from cirq.linalg import decompositions, predicates
Performs a multiplexed rotation over all qubits in this unitary matrix,

    Uses ry and rz multiplexing for quantum shannon decomposition

    Args:
        cossin_qubits: Subset of total qubits involved in this unitary gate
        angles: List of angles to be multiplexed over for the given type of rotation
        rot_func: Rotation function used for this multiplexing implementation
                    (cirq.ry or cirq.rz)

    Calls:
        No major calls

    Yields: Single operation from OP TREE from set 1- and 2-qubit gates: {ry,rz,CNOT}
    