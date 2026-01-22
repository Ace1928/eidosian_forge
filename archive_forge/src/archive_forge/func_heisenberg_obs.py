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
def heisenberg_obs(self, wire_order):
    """Representation of the observable in the position/momentum operator basis.

        Returns the expansion :math:`q` of the observable, :math:`Q`, in the
        basis :math:`\\mathbf{r} = (\\I, \\x_0, \\p_0, \\x_1, \\p_1, \\ldots)`.

        * For first-order observables returns a real vector such
          that :math:`Q = \\sum_i q_i \\mathbf{r}_i`.

        * For second-order observables returns a real symmetric matrix
          such that :math:`Q = \\sum_{ij} q_{ij} \\mathbf{r}_i \\mathbf{r}_j`.

        Args:
            wire_order (Wires): global wire order defining which subspace the operator acts on
        Returns:
            array[float]: :math:`q`
        """
    p = self.parameters
    U = self._heisenberg_rep(p)
    return self.heisenberg_expand(U, wire_order)