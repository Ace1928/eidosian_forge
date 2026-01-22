from __future__ import annotations
import itertools
from itertools import combinations
import copy
from functools import reduce
from operator import mul
import numpy as np
from qiskit.exceptions import QiskitError
def set_pj(self, indices):
    """Set to special form polynomial on subset of variables.

        p_J(x) := sum_{a subseteq J,|a| neq 0} (-2)^{|a|-1}x^a
        """
    indices_arr = np.array(indices)
    if (indices_arr < 0).any() or (indices_arr >= self.n_vars).any():
        raise QiskitError('Indices are out of bounds.')
    indices = sorted(indices)
    subsets_2 = itertools.combinations(indices, 2)
    subsets_3 = itertools.combinations(indices, 3)
    self.weight_0 = 0
    self.weight_1 = np.zeros(self.n_vars)
    self.weight_2 = np.zeros(self.nc2)
    self.weight_3 = np.zeros(self.nc3)
    for j in indices:
        self.set_term([j], 1)
    for j in subsets_2:
        self.set_term(list(j), 6)
    for j in subsets_3:
        self.set_term(list(j), 4)