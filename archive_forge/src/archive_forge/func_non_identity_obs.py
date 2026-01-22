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
@property
def non_identity_obs(self):
    """Returns the non-identity observables contained in the tensor product.

        Returns:
            list[:class:`~.Observable`]: list containing the non-identity observables
            in the tensor product
        """
    return [obs for obs in self.obs if not isinstance(obs, qml.Identity)]