from ..util.testing import requires
from .._equilibrium import equilibrium_residual, solve_equilibrium, _get_rc_interval
from .._util import prodpow
@requires('numpy')
def test_get__rc_interval():
    c = np.array((13.0, 11, 17))
    stoich = np.array((-2, 3, -4))
    limits = _get_rc_interval(stoich, c)
    lower = -11 / 3.0
    upper = 17.0 / 4
    assert abs(limits[0] - lower) < 1e-14
    assert abs(limits[1] - upper) < 1e-14