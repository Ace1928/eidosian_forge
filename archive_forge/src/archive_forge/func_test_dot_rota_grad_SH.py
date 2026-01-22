from sympy.core.numbers import (I, pi, Rational)
from sympy.core.singleton import S
from sympy.core.symbol import symbols
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.functions.special.spherical_harmonics import Ynm
from sympy.matrices.dense import Matrix
from sympy.physics.wigner import (clebsch_gordan, wigner_9j, wigner_6j, gaunt,
from sympy.testing.pytest import raises
def test_dot_rota_grad_SH():
    theta, phi = symbols('theta phi')
    assert dot_rot_grad_Ynm(1, 1, 1, 1, 1, 0) != sqrt(30) * Ynm(2, 2, 1, 0) / (10 * sqrt(pi))
    assert dot_rot_grad_Ynm(1, 1, 1, 1, 1, 0).doit() == sqrt(30) * Ynm(2, 2, 1, 0) / (10 * sqrt(pi))
    assert dot_rot_grad_Ynm(1, 5, 1, 1, 1, 2) != 0
    assert dot_rot_grad_Ynm(1, 5, 1, 1, 1, 2).doit() == 0
    assert dot_rot_grad_Ynm(3, 3, 3, 3, theta, phi).doit() == 15 * sqrt(3003) * Ynm(6, 6, theta, phi) / (143 * sqrt(pi))
    assert dot_rot_grad_Ynm(3, 3, 1, 1, theta, phi).doit() == sqrt(3) * Ynm(4, 4, theta, phi) / sqrt(pi)
    assert dot_rot_grad_Ynm(3, 2, 2, 0, theta, phi).doit() == 3 * sqrt(55) * Ynm(5, 2, theta, phi) / (11 * sqrt(pi))
    assert dot_rot_grad_Ynm(3, 2, 3, 2, theta, phi).doit().expand() == -sqrt(70) * Ynm(4, 4, theta, phi) / (11 * sqrt(pi)) + 45 * sqrt(182) * Ynm(6, 4, theta, phi) / (143 * sqrt(pi))