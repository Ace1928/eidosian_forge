from itertools import chain
import pytest
from ..util.testing import requires
from ..util.parsing import parsing_library
from ..units import default_units, units_library, allclose
from ..chemistry import Substance, Reaction
from ..reactionsystem import ReactionSystem
@requires(parsing_library)
def test_ReactionSystem__identify_equilibria():
    rsys = ReactionSystem.from_string('\n    2 H2 +  O2 -> 2 H2O     ; 1e-3\n           H2O -> H+ + OH-  ; 1e-4/55.35\n      H+ + OH- -> H2O       ; 1e10\n         2 H2O -> 2 H2 + O2\n    ')
    assert rsys.identify_equilibria() == [(0, 3), (1, 2)]