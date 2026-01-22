from sympy.concrete.summations import Sum
from sympy.core.function import expand
from sympy.core.numbers import Integer
from sympy.matrices.dense import (Matrix, eye)
from sympy.tensor.indexed import Indexed
from sympy.combinatorics import Permutation
from sympy.core import S, Rational, Symbol, Basic, Add
from sympy.core.containers import Tuple
from sympy.core.symbol import symbols
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.tensor.array import Array
from sympy.tensor.tensor import TensorIndexType, tensor_indices, TensorSymmetry, \
from sympy.testing.pytest import raises, XFAIL, warns_deprecated_sympy
from sympy.matrices import diag
def test_TensMul_data():
    with warns_deprecated_sympy():
        Lorentz = TensorIndexType('Lorentz', metric_symmetry=1, dummy_name='L', dim=4)
        Lorentz.data = [-1, 1, 1, 1]
        mu, nu, alpha, beta = tensor_indices('\\mu, \\nu, \\alpha, \\beta', Lorentz)
        u = TensorHead('u', [Lorentz])
        u.data = [1, 0, 0, 0]
        F = TensorHead('F', [Lorentz] * 2, TensorSymmetry.fully_symmetric(-2))
        Ex, Ey, Ez, Bx, By, Bz = symbols('E_x E_y E_z B_x B_y B_z')
        F.data = [[0, Ex, Ey, Ez], [-Ex, 0, Bz, -By], [-Ey, -Bz, 0, Bx], [-Ez, By, -Bx, 0]]
        E = F(mu, nu) * u(-nu)
        assert (E(mu) * E(nu)).data == Array([[0, 0, 0, 0], [0, Ex ** 2, Ex * Ey, Ex * Ez], [0, Ex * Ey, Ey ** 2, Ey * Ez], [0, Ex * Ez, Ey * Ez, Ez ** 2]])
        assert (E(mu) * E(nu)).canon_bp().data == (E(mu) * E(nu)).data
        assert (F(mu, alpha) * F(beta, nu) * u(-alpha) * u(-beta)).data == -(E(mu) * E(nu)).data
        assert (F(alpha, mu) * F(beta, nu) * u(-alpha) * u(-beta)).data == (E(mu) * E(nu)).data
        g = TensorHead('g', [Lorentz] * 2, TensorSymmetry.fully_symmetric(2))
        g.data = Lorentz.data
        perp = u(mu) * u(nu) + g(mu, nu)
        mul_1 = u(-mu) * perp(mu, nu)
        assert mul_1.data == Array([0, 0, 0, 0])
        mul_2 = u(-mu) * perp(mu, alpha) * perp(nu, beta)
        assert mul_2.data == Array.zeros(4, 4, 4)
        Fperp = perp(mu, alpha) * perp(nu, beta) * F(-alpha, -beta)
        assert Fperp.data[0, :] == Array([0, 0, 0, 0])
        assert Fperp.data[:, 0] == Array([0, 0, 0, 0])
        mul_3 = u(-mu) * Fperp(mu, nu)
        assert mul_3.data == Array([0, 0, 0, 0])
        del g.data