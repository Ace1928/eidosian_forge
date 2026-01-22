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
def to_rotation_matrix(self, v=None, homogeneous=True):
    """Returns the equivalent rotation transformation matrix of the quaternion
        which represents rotation about the origin if v is not passed.

        Parameters
        ==========

        v : tuple or None
            Default value: None
        homogeneous : bool
            When True, gives an expression that may be more efficient for
            symbolic calculations but less so for direct evaluation. Both
            formulas are mathematically equivalent.
            Default value: True

        Returns
        =======

        tuple
            Returns the equivalent rotation transformation matrix of the quaternion
            which represents rotation about the origin if v is not passed.

        Examples
        ========

        >>> from sympy import Quaternion
        >>> from sympy import symbols, trigsimp, cos, sin
        >>> x = symbols('x')
        >>> q = Quaternion(cos(x/2), 0, 0, sin(x/2))
        >>> trigsimp(q.to_rotation_matrix())
        Matrix([
        [cos(x), -sin(x), 0],
        [sin(x),  cos(x), 0],
        [     0,       0, 1]])

        Generates a 4x4 transformation matrix (used for rotation about a point
        other than the origin) if the point(v) is passed as an argument.
        """
    q = self
    s = q.norm() ** (-2)
    if homogeneous:
        m00 = s * (q.a ** 2 + q.b ** 2 - q.c ** 2 - q.d ** 2)
        m11 = s * (q.a ** 2 - q.b ** 2 + q.c ** 2 - q.d ** 2)
        m22 = s * (q.a ** 2 - q.b ** 2 - q.c ** 2 + q.d ** 2)
    else:
        m00 = 1 - 2 * s * (q.c ** 2 + q.d ** 2)
        m11 = 1 - 2 * s * (q.b ** 2 + q.d ** 2)
        m22 = 1 - 2 * s * (q.b ** 2 + q.c ** 2)
    m01 = 2 * s * (q.b * q.c - q.d * q.a)
    m02 = 2 * s * (q.b * q.d + q.c * q.a)
    m10 = 2 * s * (q.b * q.c + q.d * q.a)
    m12 = 2 * s * (q.c * q.d - q.b * q.a)
    m20 = 2 * s * (q.b * q.d - q.c * q.a)
    m21 = 2 * s * (q.c * q.d + q.b * q.a)
    if not v:
        return Matrix([[m00, m01, m02], [m10, m11, m12], [m20, m21, m22]])
    else:
        x, y, z = v
        m03 = x - x * m00 - y * m01 - z * m02
        m13 = y - x * m10 - y * m11 - z * m12
        m23 = z - x * m20 - y * m21 - z * m22
        m30 = m31 = m32 = 0
        m33 = 1
        return Matrix([[m00, m01, m02, m03], [m10, m11, m12, m13], [m20, m21, m22, m23], [m30, m31, m32, m33]])