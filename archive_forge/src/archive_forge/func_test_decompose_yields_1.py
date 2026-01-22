from chempy.units import units_library, allclose, _sum
from ..testing import requires
@requires('numpy')
def test_decompose_yields_1():
    from chempy import Reaction
    gamma_yields = {'OH-': 0.5, 'H2O2': 0.7, 'OH': 2.7, 'H2': 0.45, 'H': 0.66, 'H+': 3.1, 'HO2': 0.02, 'e-(aq)': 2.6}
    rxns = [Reaction({'H2O': 1}, {'H+': 1, 'OH-': 1}), Reaction({'H2O': 1}, {'H+': 1, 'e-(aq)': 1, 'OH': 1}), Reaction({'H2O': 1}, {'H': 2, 'H2O2': 1}, inact_reac={'H2O': 1}), Reaction({'H2O': 1}, {'H2': 1, 'H2O2': 1}, inact_reac={'H2O': 1}), Reaction({'H2O': 1}, {'H2': 1, 'OH': 2}, inact_reac={'H2O': 1}), Reaction({'H2O': 1}, {'H2': 3, 'HO2': 2}, inact_reac={'H2O': 3})]
    k = decompose_yields(gamma_yields, rxns)
    k_ref = [0.5, 2.6, 0.33, 0.37, 0.05, 0.01]
    assert np.allclose(k, k_ref)
    G_H2O = sum((rxn.net_stoich(['H2O'])[0] * k[i] for i, rxn in enumerate(rxns)))
    assert abs(G_H2O + 4.64) < 0.001