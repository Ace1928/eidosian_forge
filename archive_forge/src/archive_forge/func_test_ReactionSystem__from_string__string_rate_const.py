from itertools import chain
import pytest
from ..util.testing import requires
from ..util.parsing import parsing_library
from ..units import default_units, units_library, allclose
from ..chemistry import Substance, Reaction
from ..reactionsystem import ReactionSystem
@requires(parsing_library)
def test_ReactionSystem__from_string__string_rate_const():
    rsys = ReactionSystem.from_string("H+ + OH- -> H2O; 'kf'")
    r2, = rsys.rxns
    assert r2.reac == {'OH-': 1, 'H+': 1}
    assert r2.prod == {'H2O': 1}
    r2str = r2.string(rsys.substances, with_param=True)
    assert r2str.endswith("; 'kf'")