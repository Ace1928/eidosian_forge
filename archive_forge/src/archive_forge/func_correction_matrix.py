import time
from typing import Any, Dict, Union, Sequence, List, Tuple, TYPE_CHECKING, Optional, cast
import sympy
import numpy as np
import scipy.optimize
from cirq import circuits, ops, vis, study
from cirq._compat import proper_repr
def correction_matrix(self, qubits: Optional[Sequence['cirq.Qid']]=None) -> np.ndarray:
    """Returns a single correction matrix constructed for the given set of qubits.

        A correction matrix is the inverse of confusion matrix and can be used to apply corrections
        to observed frequencies / probabilities to compensate for the readout error.
        A Moore–Penrose Pseudo inverse of the confusion matrix is computed to get the correction
        matrix.

        Args:
            qubits: The qubits representing the subspace for which a correction matrix should be
                    constructed. By default, uses all qubits in sorted order, i.e. `self.qubits`.
                    Note that ordering of qubits sets the basis ordering of the returned matrix.

        Returns:
            Correction matrix for subspace corresponding to `qubits`.

        Raises:
            ValueError: If `qubits` is not a subset of `self.qubits`.
        """
    if qubits is None:
        qubits = self.qubits
    if any((q not in self.qubits for q in qubits)):
        raise ValueError(f'qubits {qubits} should be a subset of self.qubits {self.qubits}.')
    return np.linalg.pinv(self.confusion_matrix(qubits))