from itertools import chain
import pytest
from ..util.testing import requires
from ..util.parsing import parsing_library
from ..units import default_units, units_library, allclose
from ..chemistry import Substance, Reaction
from ..reactionsystem import ReactionSystem
def test_ReactionSystem__per_reaction_effect_on_substance():
    rs = ReactionSystem([Reaction({'H2': 2, 'O2': 1}, {'H2O': 2})])
    assert rs.per_reaction_effect_on_substance('H2') == {0: -2}
    assert rs.per_reaction_effect_on_substance('O2') == {0: -1}
    assert rs.per_reaction_effect_on_substance('H2O') == {0: 2}