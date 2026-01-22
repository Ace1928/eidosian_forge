from itertools import chain
import pytest
from ..util.testing import requires
from ..util.parsing import parsing_library
from ..units import default_units, units_library, allclose
from ..chemistry import Substance, Reaction
from ..reactionsystem import ReactionSystem
@requires(parsing_library, 'numpy')
def test_ReactionSystem__substance_factory():
    r1 = Reaction.from_string('H2O -> H+ + OH-', 'H2O H+ OH-')
    rs = ReactionSystem([r1], 'H2O H+ OH-', substance_factory=Substance.from_formula)
    assert rs.net_stoichs(['H2O']) == [-1]
    assert rs.net_stoichs(['H+']) == [1]
    assert rs.net_stoichs(['OH-']) == [1]
    assert rs.substances['H2O'].composition[8] == 1
    assert rs.substances['OH-'].composition[0] == -1
    assert rs.substances['H+'].charge == 1