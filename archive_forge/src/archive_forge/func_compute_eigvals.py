import warnings
from typing import Iterable
from functools import lru_cache
import numpy as np
from scipy.linalg import block_diag
import pennylane as qml
from pennylane.operation import (
from pennylane.ops.qubit.matrix_ops import QubitUnitary
from pennylane.ops.qubit.parametric_ops_single_qubit import stack_last
from .controlled import ControlledOp
from .controlled_decompositions import decompose_mcx
@staticmethod
def compute_eigvals(phi, **_):
    """Eigenvalues of the operator in the computational basis (static method).

        If :attr:`diagonalizing_gates` are specified and implement a unitary :math:`U^{\\dagger}`,
        the operator can be reconstructed as

        .. math:: O = U \\Sigma U^{\\dagger},

        where :math:`\\Sigma` is the diagonal matrix containing the eigenvalues.

        Otherwise, no particular order for the eigenvalues is guaranteed.

        .. seealso:: :meth:`~.ControlledPhaseShift.eigvals`


        Args:
            phi (tensor_like or float): phase shift

        Returns:
            tensor_like: eigenvalues

        **Example**

        >>> qml.ControlledPhaseShift.compute_eigvals(torch.tensor(0.5))
        tensor([1.0000+0.0000j, 1.0000+0.0000j, 1.0000+0.0000j, 0.8776+0.4794j])
        """
    if qml.math.get_interface(phi) == 'tensorflow':
        phase = qml.math.exp(1j * qml.math.cast_like(phi, 1j))
        ones = qml.math.ones_like(phase)
        return stack_last([ones, ones, ones, phase])
    prefactors = qml.math.array([0, 0, 0, 1j], like=phi)
    if qml.math.ndim(phi) == 0:
        product = phi * prefactors
    else:
        product = qml.math.outer(phi, prefactors)
    return qml.math.exp(product)