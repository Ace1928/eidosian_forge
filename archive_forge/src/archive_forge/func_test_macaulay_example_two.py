from sympy.matrices.dense import Matrix
from sympy.polys.polytools import factor
from sympy.core import symbols
from sympy.tensor.indexed import IndexedBase
from sympy.polys.multivariate_resultants import (DixonResultant,
def test_macaulay_example_two():
    """Tests the Macaulay formulation for example from [Stiller96]_."""
    x, y, z = symbols('x, y, z')
    a_0, a_1, a_2 = symbols('a_0, a_1, a_2')
    b_0, b_1, b_2 = symbols('b_0, b_1, b_2')
    c_0, c_1, c_2, c_3, c_4 = symbols('c_0, c_1, c_2, c_3, c_4')
    f = a_0 * y - a_1 * x + a_2 * z
    g = b_1 * x ** 2 + b_0 * y ** 2 - b_2 * z ** 2
    h = c_0 * y - c_1 * x ** 3 + c_2 * x ** 2 * z - c_3 * x * z ** 2 + c_4 * z ** 3
    mac = MacaulayResultant([f, g, h], [x, y, z])
    assert mac.degrees == [1, 2, 3]
    assert mac.degree_m == 4
    assert mac.monomials_size == 15
    assert len(mac.get_row_coefficients()) == mac.n
    matrix = mac.get_matrix()
    assert matrix.shape == (mac.monomials_size, mac.monomials_size)
    assert mac.get_submatrix(matrix) == Matrix([[-a_1, a_0, a_2, 0], [0, -a_1, 0, 0], [0, 0, -a_1, 0], [0, 0, 0, -a_1]])