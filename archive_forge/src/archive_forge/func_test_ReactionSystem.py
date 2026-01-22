from itertools import chain
import pytest
from ..util.testing import requires
from ..util.parsing import parsing_library
from ..units import default_units, units_library, allclose
from ..chemistry import Substance, Reaction
from ..reactionsystem import ReactionSystem
@requires(parsing_library, 'numpy')
def test_ReactionSystem():
    import numpy as np
    kw = dict(substance_factory=Substance.from_formula)
    r1 = Reaction.from_string('H2O -> H+ + OH-', 'H2O H+ OH-', name='r1')
    rs = ReactionSystem([r1], 'H2O H+ OH-', **kw)
    r2 = Reaction.from_string('H2O -> 2 H+ + OH-', 'H2O H+ OH-', name='r2')
    with pytest.raises(ValueError):
        ReactionSystem([r2], 'H2O H+ OH-', **kw)
    with pytest.raises(ValueError):
        ReactionSystem([r1, r1], 'H2O H+ OH-', **kw)
    assert rs.as_substance_index('H2O') == 0
    assert rs.as_substance_index(0) == 0
    varied, varied_keys = rs.per_substance_varied({'H2O': 55.4, 'H+': 1e-07, 'OH-': 1e-07}, {'H+': [1e-08, 1e-09, 1e-10, 1e-11], 'OH-': [0.001, 0.01]})
    assert varied_keys == ('H+', 'OH-')
    assert len(varied.shape) == 3
    assert varied.shape[:-1] == (4, 2)
    assert varied.shape[-1] == 3
    assert np.all(varied[..., 0] == 55.4)
    assert np.all(varied[:, 1, 2] == 0.01)
    assert rs['r1'] is r1
    rs.rxns.append(r2)
    assert rs['r2'] is r2
    with pytest.raises(KeyError):
        rs['r3']
    rs.rxns.append(Reaction({}, {}, 0, name='r2', checks=()))
    with pytest.raises(ValueError):
        rs['r2']
    empty_rs = ReactionSystem([])
    rs2 = empty_rs + rs
    assert rs2 == rs
    rs3 = rs + empty_rs
    assert rs3 == rs