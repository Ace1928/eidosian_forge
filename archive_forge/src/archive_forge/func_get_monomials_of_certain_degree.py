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
def get_monomials_of_certain_degree(self, degree):
    """
        Returns
        =======

        monomials: list
            A list of monomials of a certain degree.
        """
    monomials = [Mul(*monomial) for monomial in combinations_with_replacement(self.variables, degree)]
    return sorted(monomials, reverse=True, key=monomial_key('lex', self.variables))