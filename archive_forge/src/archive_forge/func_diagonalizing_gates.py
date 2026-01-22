import abc
from typing import Callable, List
import copy
import pennylane as qml
from pennylane import math
from pennylane.operation import Operator, _UNSET_BATCH_SIZE
from pennylane.wires import Wires
def diagonalizing_gates(self):
    """Sequence of gates that diagonalize the operator in the computational basis.

        Given the eigendecomposition :math:`O = U \\Sigma U^{\\dagger}` where
        :math:`\\Sigma` is a diagonal matrix containing the eigenvalues,
        the sequence of diagonalizing gates implements the unitary :math:`U^{\\dagger}`.

        The diagonalizing gates rotate the state into the eigenbasis
        of the operator.

        A ``DiagGatesUndefinedError`` is raised if no representation by decomposition is defined.

        .. seealso:: :meth:`~.Operator.compute_diagonalizing_gates`.

        Returns:
            list[.Operator] or None: a list of operators
        """
    diag_gates = []
    for ops in self.overlapping_ops:
        if len(ops) == 1:
            diag_gates.extend(ops[0].diagonalizing_gates())
        else:
            tmp_sum = self.__class__(*ops)
            eigvecs = tmp_sum.eigendecomposition['eigvec']
            diag_gates.append(qml.QubitUnitary(math.transpose(math.conj(eigvecs)), wires=tmp_sum.wires))
    return diag_gates