from math import prod
from sympy.core.mul import Mul
from sympy.matrices.dense import (Matrix, diag)
from sympy.polys.polytools import (Poly, degree_list, rem)
from sympy.simplify.simplify import simplify
from sympy.tensor.indexed import IndexedBase
from sympy.polys.monomials import itermonomials, monomial_deg
from sympy.polys.orderings import monomial_key
from sympy.polys.polytools import poly_from_expr, total_degree
from sympy.functions.combinatorial.factorials import binomial
from itertools import combinations_with_replacement
from sympy.utilities.exceptions import sympy_deprecation_warning
def get_reduced_nonreduced(self):
    """
        Returns
        =======

        reduced: list
            A list of the reduced monomials
        non_reduced: list
            A list of the monomials that are not reduced

        Definition
        ==========

        A polynomial is said to be reduced in x_i, if its degree (the
        maximum degree of its monomials) in x_i is less than d_i. A
        polynomial that is reduced in all variables but one is said
        simply to be reduced.
        """
    divisible = []
    for m in self.monomial_set:
        temp = []
        for i, v in enumerate(self.variables):
            temp.append(bool(total_degree(m, v) >= self.degrees[i]))
        divisible.append(temp)
    reduced = [i for i, r in enumerate(divisible) if sum(r) < self.n - 1]
    non_reduced = [i for i, r in enumerate(divisible) if sum(r) >= self.n - 1]
    return (reduced, non_reduced)