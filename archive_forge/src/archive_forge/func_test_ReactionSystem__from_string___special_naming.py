from itertools import chain
import pytest
from ..util.testing import requires
from ..util.parsing import parsing_library
from ..units import default_units, units_library, allclose
from ..chemistry import Substance, Reaction
from ..reactionsystem import ReactionSystem
@requires(parsing_library, 'numpy')
def test_ReactionSystem__from_string___special_naming():
    rs = ReactionSystem.from_string('\nH2O* + H2O -> 2 H2O\nH2O* -> OH + H\n')
    for sk in 'H2O* H2O OH H'.split():
        assert sk in rs.substances
    assert rs.substances['H2O*'].composition == {1: 2, 8: 1}
    assert rs.categorize_substances() == dict(accumulated={'OH', 'H', 'H2O'}, depleted={'H2O*'}, unaffected=set(), nonparticipating=set())