from __future__ import annotations
import itertools
from itertools import combinations
import copy
from functools import reduce
from operator import mul
import numpy as np
from qiskit.exceptions import QiskitError
def mul_monomial(self, indices):
    """Multiply by a monomial given by indices.

        Returns the product.
        """
    length = len(indices)
    if length >= 4:
        raise QiskitError('There is no term with on more than 3 indices.')
    indices_arr = np.array(indices)
    if (indices_arr < 0).any() and (indices_arr > self.n_vars).any():
        raise QiskitError('Indices are out of bounds.')
    if length > 1 and (np.diff(indices_arr) <= 0).any():
        raise QiskitError('Indices are non-increasing!')
    result = SpecialPolynomial(self.n_vars)
    if length == 0:
        result = copy.deepcopy(self)
    else:
        terms0 = [[]]
        terms1 = list(combinations(range(self.n_vars), r=1))
        terms2 = list(combinations(range(self.n_vars), r=2))
        terms3 = list(combinations(range(self.n_vars), r=3))
        for term in terms0 + terms1 + terms2 + terms3:
            value = self.get_term(term)
            new_term = list(set(term).union(set(indices)))
            result.set_term(new_term, (result.get_term(new_term) + value) % 8)
    return result