from sympy.core import S, Rational
from sympy.integrals.quadrature import (gauss_legendre, gauss_laguerre,
def test_chebyshev_u():
    x, w = gauss_chebyshev_u(1, 17)
    assert [str(r) for r in x] == ['0']
    assert [str(r) for r in w] == ['1.5707963267948966']
    x, w = gauss_chebyshev_u(2, 17)
    assert [str(r) for r in x] == ['0.50000000000000000', '-0.50000000000000000']
    assert [str(r) for r in w] == ['0.78539816339744831', '0.78539816339744831']
    x, w = gauss_chebyshev_u(3, 17)
    assert [str(r) for r in x] == ['0.70710678118654752', '0', '-0.70710678118654752']
    assert [str(r) for r in w] == ['0.39269908169872415', '0.78539816339744831', '0.39269908169872415']
    x, w = gauss_chebyshev_u(4, 17)
    assert [str(r) for r in x] == ['0.80901699437494742', '0.30901699437494742', '-0.30901699437494742', '-0.80901699437494742']
    assert [str(r) for r in w] == ['0.21707871342270599', '0.56831944997474231', '0.56831944997474231', '0.21707871342270599']
    x, w = gauss_chebyshev_u(5, 17)
    assert [str(r) for r in x] == ['0.86602540378443865', '0.50000000000000000', '0', '-0.50000000000000000', '-0.86602540378443865']
    assert [str(r) for r in w] == ['0.13089969389957472', '0.39269908169872415', '0.52359877559829887', '0.39269908169872415', '0.13089969389957472']