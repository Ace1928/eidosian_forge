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
def get_row_coefficients(self):
    """
        Returns
        =======

        row_coefficients: list
            The row coefficients of Macaulay's matrix
        """
    row_coefficients = []
    divisible = []
    for i in range(self.n):
        if i == 0:
            degree = self.degree_m - self.degrees[i]
            monomial = self.get_monomials_of_certain_degree(degree)
            row_coefficients.append(monomial)
        else:
            divisible.append(self.variables[i - 1] ** self.degrees[i - 1])
            degree = self.degree_m - self.degrees[i]
            poss_rows = self.get_monomials_of_certain_degree(degree)
            for div in divisible:
                for p in poss_rows:
                    if rem(p, div) == 0:
                        poss_rows = [item for item in poss_rows if item != p]
            row_coefficients.append(poss_rows)
    return row_coefficients