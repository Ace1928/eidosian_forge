from __future__ import annotations
import itertools
from itertools import combinations
import copy
from functools import reduce
from operator import mul
import numpy as np
from qiskit.exceptions import QiskitError
def set_term(self, indices, value):
    """Set the value of a term given the list of variables.

        Example: indices = [] returns the constant
                 indices = [0] returns the coefficient of x_0
                 indices = [0,3] returns the coefficient of x_0x_3
                 indices = [0,1,3] returns the coefficient of x_0x_1x_3

        If len(indices) > 3 the method fails.
        If the indices are out of bounds the method fails.
        If the indices are not increasing the method fails.
        The value is reduced modulo 8.
        """
    length = len(indices)
    if length >= 4:
        return
    indices_arr = np.array(indices)
    if (indices_arr < 0).any() or (indices_arr >= self.n_vars).any():
        raise QiskitError('Indices are out of bounds.')
    if length > 1 and (np.diff(indices_arr) <= 0).any():
        raise QiskitError('Indices are non-increasing.')
    value = value % 8
    if length == 0:
        self.weight_0 = value
    elif length == 1:
        self.weight_1[indices[0]] = value
    elif length == 2:
        offset_1 = int(indices[0] * self.n_vars - (indices[0] + 1) * indices[0] / 2)
        offset_2 = int(indices[1] - indices[0] - 1)
        self.weight_2[offset_1 + offset_2] = value
    else:
        tmp_1 = self.n_vars - indices[0]
        offset_1 = int((tmp_1 - 3) * (tmp_1 - 2) * (tmp_1 - 1) / 6)
        tmp_2 = self.n_vars - indices[1]
        offset_2 = int((tmp_2 - 2) * (tmp_2 - 1) / 2)
        offset_3 = self.n_vars - indices[2]
        offset = int(self.n_vars * (self.n_vars - 1) * (self.n_vars - 2) / 6 - offset_1 - offset_2 - offset_3)
        self.weight_3[offset] = value