from math import log as ln
from ..electrolytes import A as A_dh, B as B_dh
from ..electrolytes import limiting_log_gamma, _ActivityProductBase, ionic_strength
from ..units import (
from ..util.testing import requires
def test_limiting_log_gamma():
    A20 = A_dh(80.1, 293.15, 998.2071) / ln(10)
    log_gamma = limiting_log_gamma(0.4, -3, A20)
    assert abs(log_gamma + 2.88413) < 0.0001