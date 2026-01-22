from sympy.core.numbers import I
from sympy.matrices.dense import MutableDenseMatrix as Matrix
from sympy.utilities.decorator import deprecated
def pat_matrix(m, dx, dy, dz):
    """Returns the Parallel Axis Theorem matrix to translate the inertia
    matrix a distance of `(dx, dy, dz)` for a body of mass m.

    Examples
    ========

    To translate a body having a mass of 2 units a distance of 1 unit along
    the `x`-axis we get:

    >>> from sympy.physics.matrices import pat_matrix
    >>> pat_matrix(2, 1, 0, 0)
    Matrix([
    [0, 0, 0],
    [0, 2, 0],
    [0, 0, 2]])

    """
    dxdy = -dx * dy
    dydz = -dy * dz
    dzdx = -dz * dx
    dxdx = dx ** 2
    dydy = dy ** 2
    dzdz = dz ** 2
    mat = ((dydy + dzdz, dxdy, dzdx), (dxdy, dxdx + dzdz, dydz), (dzdx, dydz, dydy + dxdx))
    return m * Matrix(mat)