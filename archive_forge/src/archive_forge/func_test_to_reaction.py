import pytest
from pyparsing import ParseException
from ..parsing import (
from ..testing import requires
@requires(parsing_library)
def test_to_reaction():
    from chempy.chemistry import Reaction, Equilibrium
    rxn = to_reaction("H+ + OH- -> H2O; 1.4e11; ref={'doi': '10.1039/FT9908601539'}", 'H+ OH- H2O'.split(), '->', Reaction)
    assert rxn.__class__ == Reaction
    assert rxn.reac['H+'] == 1
    assert rxn.reac['OH-'] == 1
    assert rxn.prod['H2O'] == 1
    assert rxn.param == 140000000000.0
    assert rxn.ref['doi'].startswith('10.')
    eq = to_reaction("H+ + OH- = H2O; 1e-14; ref='rt, [H2O] == 1 M'", 'H+ OH- H2O'.split(), '=', Equilibrium)
    assert eq.__class__ == Equilibrium
    assert eq.reac['H+'] == 1
    assert eq.reac['OH-'] == 1
    assert eq.prod['H2O'] == 1
    assert eq.ref.startswith('rt')
    for s in ['2 e-(aq) + (2 H2O) -> H2 + 2 OH- ; 1e6 ; ', '2 * e-(aq) + (2 H2O) -> 1 * H2 + 2 * OH- ; 1e6 ; ']:
        rxn2 = to_reaction(s, 'e-(aq) H2 OH- H2O'.split(), '->', Reaction)
        assert rxn2.__class__ == Reaction
        assert rxn2.reac['e-(aq)'] == 2
        assert rxn2.inact_reac['H2O'] == 2
        assert rxn2.prod['H2'] == 1
        assert rxn2.prod['OH-'] == 2
        assert rxn2.param == 1000000.0
    r1 = to_reaction('-> H2O', None, '->', Reaction)
    assert r1.reac == {}
    assert r1.prod == {'H2O': 1}
    assert r1.param is None
    r2 = to_reaction('H2O ->', None, '->', Reaction)
    assert r2.reac == {'H2O': 1}
    assert r2.prod == {}
    assert r2.param is None
    from chempy.kinetics.rates import MassAction
    ma = MassAction([3.14])
    r3 = to_reaction('H+ + OH- -> H2O', None, '->', Reaction, param=ma)
    assert r3.param.args == [3.14]
    rxn3 = to_reaction('H2O + H2O -> H3O+ + OH-', 'H3O+ OH- H2O'.split(), '->', Reaction)
    assert rxn3.reac == {'H2O': 2} and rxn3.prod == {'H3O+': 1, 'OH-': 1}
    rxn4 = to_reaction('2 e-(aq) + (2 H2O) + (2 H+) -> H2 + 2 H2O', 'e-(aq) H2 H2O H+'.split(), '->', Reaction)
    assert rxn4.reac == {'e-(aq)': 2} and rxn4.inact_reac == {'H2O': 2, 'H+': 2} and (rxn4.prod == {'H2': 1, 'H2O': 2})