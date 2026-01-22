from __future__ import annotations
import typing
from abc import ABC
import numpy as np
from numpy import linalg as la
from .approximate import ApproximatingObjective
from .elementary_operations import ry_matrix, rz_matrix, place_unitary, place_cnot, rx_matrix
@property
def num_cnots(self):
    """
        Returns:
            A number of CNOT units to be used by the approximate circuit.
        """
    return self._num_cnots