from itertools import chain
import pytest
from ..util.testing import requires
from ..util.parsing import parsing_library
from ..units import default_units, units_library, allclose
from ..chemistry import Substance, Reaction
from ..reactionsystem import ReactionSystem
def test_ReactionSystem__rates__cstr():
    k = 11
    rs = ReactionSystem([Reaction({'H2O2': 2}, {'O2': 1, 'H2O': 2}, k)])
    c0 = {'H2O2': 3, 'O2': 5, 'H2O': 53}
    fr = 7
    fc = {'H2O2': 13, 'O2': 17, 'H2O': 23}
    r = k * c0['H2O2'] ** 2
    ref = {'H2O2': -2 * r + fr * fc['H2O2'] - fr * c0['H2O2'], 'O2': r + fr * fc['O2'] - fr * c0['O2'], 'H2O': 2 * r + fr * fc['H2O'] - fr * c0['H2O']}
    variables = dict(chain(c0.items(), [('fc_' + key, val) for key, val in fc.items()], [('fr', fr)]))
    assert rs.rates(variables, cstr_fr_fc=('fr', {sk: 'fc_' + sk for sk in rs.substances})) == ref