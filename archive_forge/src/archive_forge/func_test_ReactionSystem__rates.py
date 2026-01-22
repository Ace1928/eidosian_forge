from itertools import chain
import pytest
from ..util.testing import requires
from ..util.parsing import parsing_library
from ..units import default_units, units_library, allclose
from ..chemistry import Substance, Reaction
from ..reactionsystem import ReactionSystem
def test_ReactionSystem__rates():
    rs = ReactionSystem([Reaction({'H2O'}, {'H+', 'OH-'}, 11)])
    assert rs.rates({'H2O': 3, 'H+': 5, 'OH-': 7}) == {'H2O': -11 * 3, 'H+': 11 * 3, 'OH-': 11 * 3}