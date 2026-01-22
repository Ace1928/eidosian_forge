from typing import Optional, List, Tuple, Iterable, Callable, Union, Dict
import numpy as np
from qiskit.exceptions import QiskitError
from ..distributions.quasi import QuasiDistribution
from ..counts import Counts
from .base_readout_mitigator import BaseReadoutMitigator
from .utils import counts_probability_vector, z_diagonal, str2diag
def quasi_probabilities(self, data: Counts, qubits: Optional[List[int]]=None, clbits: Optional[List[int]]=None, shots: Optional[int]=None) -> QuasiDistribution:
    """Compute mitigated quasi probabilities value.

        Args:
            data: counts object
            qubits: qubits the count bitstrings correspond to.
            clbits: Optional, marginalize counts to just these bits.
            shots: Optional, the total number of shots, if None shots will
                be calculated as the sum of all counts.

        Returns:
            QuasiDistribution: A dictionary containing pairs of [output, mean] where "output"
                is the key in the dictionaries,
                which is the length-N bitstring of a measured standard basis state,
                and "mean" is the mean of non-zero quasi-probability estimates.

        Raises:
            QiskitError: if qubit and clbit kwargs are not valid.
        """
    if qubits is None:
        qubits = self._qubits
    num_qubits = len(qubits)
    probs_vec, calculated_shots = counts_probability_vector(data, qubit_index=self._qubit_index, clbits=clbits, qubits=qubits)
    if shots is None:
        shots = calculated_shots
    qubit_indices = [self._qubit_index[qubit] for qubit in qubits]
    ainvs = self._mitigation_mats[qubit_indices]
    prob_tens = np.reshape(probs_vec, num_qubits * [2])
    einsum_args = [prob_tens, list(range(num_qubits))]
    for i, ainv in enumerate(reversed(ainvs)):
        einsum_args += [ainv, [num_qubits + i, i]]
    einsum_args += [list(range(num_qubits, 2 * num_qubits))]
    probs_vec = np.einsum(*einsum_args).ravel()
    probs_dict = {}
    for index, _ in enumerate(probs_vec):
        probs_dict[index] = probs_vec[index]
    quasi_dist = QuasiDistribution(probs_dict, shots=shots, stddev_upper_bound=self.stddev_upper_bound(shots, qubits))
    return quasi_dist