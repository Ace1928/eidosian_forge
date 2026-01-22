from chempy.units import units_library, allclose, _sum
from ..testing import requires
@requires('numpy')
def test_decompose_yields_2():
    from chempy import Reaction
    yields = {'B': 3.0, 'C': 24.0}
    rxns = [Reaction({'A': 1}, {'B': 1, 'C': 1}, inact_reac={'A': 1}), Reaction({'A': 1}, {'C': 3})]
    k = decompose_yields(yields, rxns)
    k_ref = [3, 7]
    rtol = 1e-12
    for a, b in zip(k, k_ref):
        assert abs(a - b) < abs(a * rtol)