import random
from sympy.core.basic import Basic
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.core.sympify import sympify
from sympy.functions.elementary.trigonometric import cos, sin
from sympy.utilities.decorator import doctest_depends_on
from sympy.utilities.exceptions import sympy_deprecation_warning
from sympy.utilities.iterables import is_sequence
from .common import ShapeError
from .decompositions import _cholesky, _LDLdecomposition
from .matrices import MatrixBase
from .repmatrix import MutableRepMatrix, RepMatrix
from .solvers import _lower_triangular_solve, _upper_triangular_solve
def rot_givens(i, j, theta, dim=3):
    """Returns a a Givens rotation matrix, a a rotation in the
    plane spanned by two coordinates axes.

    Explanation
    ===========

    The Givens rotation corresponds to a generalization of rotation
    matrices to any number of dimensions, given by:

    .. math::
        G(i, j, \\theta) =
            \\begin{bmatrix}
                1   & \\cdots &    0   & \\cdots &    0   & \\cdots &    0   \\\\
                \\vdots & \\ddots & \\vdots &        & \\vdots &        & \\vdots \\\\
                0   & \\cdots &    c   & \\cdots &   -s   & \\cdots &    0   \\\\
                \\vdots &        & \\vdots & \\ddots & \\vdots &        & \\vdots \\\\
                0   & \\cdots &    s   & \\cdots &    c   & \\cdots &    0   \\\\
                \\vdots &        & \\vdots &        & \\vdots & \\ddots & \\vdots \\\\
                0   & \\cdots &    0   & \\cdots &    0   & \\cdots &    1
            \\end{bmatrix}

    Where $c = \\cos(\\theta)$ and $s = \\sin(\\theta)$ appear at the intersections
    ``i``\\th and ``j``\\th rows and columns.

    For fixed ``i > j``\\, the non-zero elements of a Givens matrix are
    given by:

    - $g_{kk} = 1$ for $k \\ne i,\\,j$
    - $g_{kk} = c$ for $k = i,\\,j$
    - $g_{ji} = -g_{ij} = -s$

    Parameters
    ==========

    i : int between ``0`` and ``dim - 1``
        Represents first axis
    j : int between ``0`` and ``dim - 1``
        Represents second axis
    dim : int bigger than 1
        Number of dimentions. Defaults to 3.

    Examples
    ========

    >>> from sympy import pi, rot_givens

    A counterclockwise rotation of pi/3 (60 degrees) around
    the third axis (z-axis):

    >>> rot_givens(1, 0, pi/3)
    Matrix([
    [      1/2, -sqrt(3)/2, 0],
    [sqrt(3)/2,        1/2, 0],
    [        0,          0, 1]])

    If we rotate by pi/2 (90 degrees):

    >>> rot_givens(1, 0, pi/2)
    Matrix([
    [0, -1, 0],
    [1,  0, 0],
    [0,  0, 1]])

    This can be generalized to any number
    of dimensions:

    >>> rot_givens(1, 0, pi/2, dim=4)
    Matrix([
    [0, -1, 0, 0],
    [1,  0, 0, 0],
    [0,  0, 1, 0],
    [0,  0, 0, 1]])

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Givens_rotation

    See Also
    ========

    rot_axis1: Returns a rotation matrix for a rotation of theta (in radians)
        about the 1-axis (clockwise around the x axis)
    rot_axis2: Returns a rotation matrix for a rotation of theta (in radians)
        about the 2-axis (clockwise around the y axis)
    rot_axis3: Returns a rotation matrix for a rotation of theta (in radians)
        about the 3-axis (clockwise around the z axis)
    rot_ccw_axis1: Returns a rotation matrix for a rotation of theta (in radians)
        about the 1-axis (counterclockwise around the x axis)
    rot_ccw_axis2: Returns a rotation matrix for a rotation of theta (in radians)
        about the 2-axis (counterclockwise around the y axis)
    rot_ccw_axis3: Returns a rotation matrix for a rotation of theta (in radians)
        about the 3-axis (counterclockwise around the z axis)
    """
    if not isinstance(dim, int) or dim < 2:
        raise ValueError('dim must be an integer biggen than one, got {}.'.format(dim))
    if i == j:
        raise ValueError('i and j must be different, got ({}, {})'.format(i, j))
    for ij in [i, j]:
        if not isinstance(ij, int) or ij < 0 or ij > dim - 1:
            raise ValueError('i and j must be integers between 0 and {}, got i={} and j={}.'.format(dim - 1, i, j))
    theta = sympify(theta)
    c = cos(theta)
    s = sin(theta)
    M = eye(dim)
    M[i, i] = c
    M[j, j] = c
    M[i, j] = s
    M[j, i] = -s
    return M