from sympy.core.numbers import (I, pi)
from sympy.functions.elementary.complexes import (Abs, im, re)
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.matrices.dense import Matrix
from sympy.simplify.simplify import simplify
from sympy.physics.quantum import TensorProduct
def mueller_matrix(J):
    """The Mueller matrix corresponding to Jones matrix `J`.

    Parameters
    ==========

    J : SymPy Matrix
        A Jones matrix.

    Returns
    =======

    SymPy Matrix
        The corresponding Mueller matrix.

    Examples
    ========

    Generic optical components.

    >>> from sympy import pprint, symbols
    >>> from sympy.physics.optics.polarization import (mueller_matrix,
    ...     linear_polarizer, half_wave_retarder, quarter_wave_retarder)
    >>> theta = symbols("theta", real=True)

    A linear_polarizer

    >>> pprint(mueller_matrix(linear_polarizer(theta)), use_unicode=True)
    ⎡            cos(2⋅θ)      sin(2⋅θ)     ⎤
    ⎢  1/2       ────────      ────────    0⎥
    ⎢               2             2         ⎥
    ⎢                                       ⎥
    ⎢cos(2⋅θ)  cos(4⋅θ)   1    sin(4⋅θ)     ⎥
    ⎢────────  ──────── + ─    ────────    0⎥
    ⎢   2         4       4       4         ⎥
    ⎢                                       ⎥
    ⎢sin(2⋅θ)    sin(4⋅θ)    1   cos(4⋅θ)   ⎥
    ⎢────────    ────────    ─ - ────────  0⎥
    ⎢   2           4        4      4       ⎥
    ⎢                                       ⎥
    ⎣   0           0             0        0⎦

    A half-wave plate

    >>> pprint(mueller_matrix(half_wave_retarder(theta)), use_unicode=True)
    ⎡1              0                           0               0 ⎤
    ⎢                                                             ⎥
    ⎢        4           2                                        ⎥
    ⎢0  8⋅sin (θ) - 8⋅sin (θ) + 1           sin(4⋅θ)            0 ⎥
    ⎢                                                             ⎥
    ⎢                                     4           2           ⎥
    ⎢0          sin(4⋅θ)           - 8⋅sin (θ) + 8⋅sin (θ) - 1  0 ⎥
    ⎢                                                             ⎥
    ⎣0              0                           0               -1⎦

    A quarter-wave plate

    >>> pprint(mueller_matrix(quarter_wave_retarder(theta)), use_unicode=True)
    ⎡1       0             0            0    ⎤
    ⎢                                        ⎥
    ⎢   cos(4⋅θ)   1    sin(4⋅θ)             ⎥
    ⎢0  ──────── + ─    ────────    -sin(2⋅θ)⎥
    ⎢      2       2       2                 ⎥
    ⎢                                        ⎥
    ⎢     sin(4⋅θ)    1   cos(4⋅θ)           ⎥
    ⎢0    ────────    ─ - ────────  cos(2⋅θ) ⎥
    ⎢        2        2      2               ⎥
    ⎢                                        ⎥
    ⎣0    sin(2⋅θ)     -cos(2⋅θ)        0    ⎦

    """
    A = Matrix([[1, 0, 0, 1], [1, 0, 0, -1], [0, 1, 1, 0], [0, -I, I, 0]])
    return simplify(A * TensorProduct(J, J.conjugate()) * A.inv())