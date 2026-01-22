from sympy.core.numbers import Rational
from sympy.core.singleton import S
from sympy.core.relational import is_eq
from sympy.functions.elementary.complexes import (conjugate, im, re, sign)
from sympy.functions.elementary.exponential import (exp, log as ln)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (acos, asin, atan2)
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.simplify.trigsimp import trigsimp
from sympy.integrals.integrals import integrate
from sympy.matrices.dense import MutableDenseMatrix as Matrix
from sympy.core.sympify import sympify, _sympify
from sympy.core.expr import Expr
from sympy.core.logic import fuzzy_not, fuzzy_or
from mpmath.libmp.libmpf import prec_to_dps
@property
def product_matrix_left(self):
    """Returns 4 x 4 Matrix equivalent to a Hamilton product from the
        left. This can be useful when treating quaternion elements as column
        vectors. Given a quaternion $q = a + bi + cj + dk$ where a, b, c and d
        are real numbers, the product matrix from the left is:

        .. math::

            M  =  \\begin{bmatrix} a  &-b  &-c  &-d \\\\
                                  b  & a  &-d  & c \\\\
                                  c  & d  & a  &-b \\\\
                                  d  &-c  & b  & a \\end{bmatrix}

        Examples
        ========

        >>> from sympy import Quaternion
        >>> from sympy.abc import a, b, c, d
        >>> q1 = Quaternion(1, 0, 0, 1)
        >>> q2 = Quaternion(a, b, c, d)
        >>> q1.product_matrix_left
        Matrix([
        [1, 0,  0, -1],
        [0, 1, -1,  0],
        [0, 1,  1,  0],
        [1, 0,  0,  1]])

        >>> q1.product_matrix_left * q2.to_Matrix()
        Matrix([
        [a - d],
        [b - c],
        [b + c],
        [a + d]])

        This is equivalent to:

        >>> (q1 * q2).to_Matrix()
        Matrix([
        [a - d],
        [b - c],
        [b + c],
        [a + d]])
        """
    return Matrix([[self.a, -self.b, -self.c, -self.d], [self.b, self.a, -self.d, self.c], [self.c, self.d, self.a, -self.b], [self.d, -self.c, self.b, self.a]])