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
def delete_zero_rows_and_columns(self, matrix):
    """Remove the zero rows and columns of the matrix."""
    rows = [i for i in range(matrix.rows) if not matrix.row(i).is_zero_matrix]
    cols = [j for j in range(matrix.cols) if not matrix.col(j).is_zero_matrix]
    return matrix[rows, cols]