import abc
import copy
import functools
import itertools
import warnings
from enum import IntEnum
from typing import List
import numpy as np
from numpy.linalg import multi_dot
from scipy.sparse import coo_matrix, eye, kron
import pennylane as qml
from pennylane.math import expand_matrix
from pennylane.queuing import QueuingManager
from pennylane.typing import TensorLike
from pennylane.wires import Wires
from .utils import pauli_eigs
from .pytrees import register_pytree
def operation_derivative(operation) -> np.ndarray:
    """Calculate the derivative of an operation.

    For an operation :math:`e^{i \\hat{H} \\phi t}`, this function returns the matrix representation
    in the standard basis of its derivative with respect to :math:`t`, i.e.,

    .. math:: \\frac{d \\, e^{i \\hat{H} \\phi t}}{dt} = i \\phi \\hat{H} e^{i \\hat{H} \\phi t},

    where :math:`\\phi` is a real constant.

    Args:
        operation (.Operation): The operation to be differentiated.

    Returns:
        array: the derivative of the operation as a matrix in the standard basis

    Raises:
        ValueError: if the operation does not have a generator or is not composed of a single
            trainable parameter
    """
    generator = qml.matrix(qml.generator(operation, format='observable'), wire_order=operation.wires)
    return 1j * generator @ operation.matrix()