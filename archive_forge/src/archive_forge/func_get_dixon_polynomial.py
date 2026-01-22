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
def get_dixon_polynomial(self):
    """
        Returns
        =======

        dixon_polynomial: polynomial
            Dixon's polynomial is calculated as:

            delta = Delta(A) / ((x_1 - a_1) ... (x_n - a_n)) where,

            A =  |p_1(x_1,... x_n), ..., p_n(x_1,... x_n)|
                 |p_1(a_1,... x_n), ..., p_n(a_1,... x_n)|
                 |...             , ...,              ...|
                 |p_1(a_1,... a_n), ..., p_n(a_1,... a_n)|
        """
    if self.m != self.n + 1:
        raise ValueError('Method invalid for given combination.')
    rows = [self.polynomials]
    temp = list(self.variables)
    for idx in range(self.n):
        temp[idx] = self.dummy_variables[idx]
        substitution = {var: t for var, t in zip(self.variables, temp)}
        rows.append([f.subs(substitution) for f in self.polynomials])
    A = Matrix(rows)
    terms = zip(self.variables, self.dummy_variables)
    product_of_differences = Mul(*[a - b for a, b in terms])
    dixon_polynomial = (A.det() / product_of_differences).factor()
    return poly_from_expr(dixon_polynomial, self.dummy_variables)[0]