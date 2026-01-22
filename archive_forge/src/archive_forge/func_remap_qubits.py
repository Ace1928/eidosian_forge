import logging
from typing import Optional, List, Tuple, Dict
import numpy as np
from qiskit.exceptions import QiskitError
from ..utils import marginal_counts
from ..counts import Counts
def remap_qubits(vec: np.ndarray, num_qubits: int, qubits: Optional[List[int]]=None) -> np.ndarray:
    """Remapping the qubits"""
    if qubits is not None:
        if len(qubits) != num_qubits:
            raise QiskitError('Num qubits does not match vector length.')
        axes = [num_qubits - 1 - i for i in reversed(np.argsort(qubits))]
        vec = np.reshape(vec, num_qubits * [2]).transpose(axes).reshape(vec.shape)
    return vec