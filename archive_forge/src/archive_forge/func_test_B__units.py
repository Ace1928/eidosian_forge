from math import log as ln
from ..electrolytes import A as A_dh, B as B_dh
from ..electrolytes import limiting_log_gamma, _ActivityProductBase, ionic_strength
from ..units import (
from ..util.testing import requires
@requires(units_library)
def test_B__units():
    B20q = B_dh(80.1, 293.15 * u.K, 998.2071 * u.kg / u.m ** 3, b0=u.mol / u.kg, constants=consts, units=u)
    close = allclose(B20q.simplified, 0.3282 / u.angstrom, rtol=0.001)
    assert close