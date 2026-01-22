import datetime
import functools
from typing import Dict, List, NamedTuple, Optional, Tuple, TYPE_CHECKING
from cirq.protocols.json_serialization import ObjectFactory
def two_qubit_matrix_gate(matrix):
    if not isinstance(matrix, np.ndarray):
        matrix = np.array(matrix, dtype=np.complex128)
    return cirq.MatrixGate(matrix, qid_shape=(2, 2))