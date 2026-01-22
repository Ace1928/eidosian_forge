import logging
from typing import Optional, List, Tuple, Dict
import numpy as np
from qiskit.exceptions import QiskitError
from ..utils import marginal_counts
from ..counts import Counts
def counts_probability_vector(counts: Counts, qubit_index: Dict[int, int], qubits: Optional[List[int]]=None, clbits: Optional[List[int]]=None) -> Tuple[np.ndarray, int]:
    """Compute a probability vector for all count outcomes.

    Args:
        counts: counts object
        qubit_index: For each qubit, its index in the mitigator qubits list
        qubits: qubits the count bitstrings correspond to.
        clbits: Optional, marginalize counts to just these bits.

    Raises:
        QiskitError: if qubits and clbits kwargs are not valid.

    Returns:
        np.ndarray: a probability vector for all count outcomes.
        int: Number of shots in the counts
    """
    counts = marganalize_counts(counts, qubit_index, qubits, clbits)
    if qubits is not None:
        num_qubits = len(qubits)
    else:
        num_qubits = len(qubit_index.keys())
    vec, shots = counts_to_vector(counts, num_qubits)
    vec = remap_qubits(vec, num_qubits, qubits)
    return (vec, shots)