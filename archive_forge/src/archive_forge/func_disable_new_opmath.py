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
def disable_new_opmath():
    """
    Change dunder methods to return Hamiltonians and Tensors instead of arithmetic operators

    **Example**

    >>> qml.operation.active_new_opmath()
    True
    >>> type(qml.X(0) @ qml.Z(1))
    <class 'pennylane.ops.op_math.prod.Prod'>
    >>> qml.operation.disable_new_opmath()
    >>> type(qml.X(0) @ qml.Z(1))
    <class 'pennylane.operation.Tensor'>
    """
    global __use_new_opmath
    __use_new_opmath = False