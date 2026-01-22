from itertools import chain
import pytest
from ..util.testing import requires
from ..util.parsing import parsing_library
from ..units import default_units, units_library, allclose
from ..chemistry import Substance, Reaction
from ..reactionsystem import ReactionSystem
@requires(parsing_library)
def test_ReactionSystem__check_balance():
    rs1 = ReactionSystem.from_string('\n'.join(['2 NH3 -> N2 + 3 H2', 'N2H4 -> N2 + 2 H2']))
    assert rs1.check_balance(strict=True)
    rs2 = ReactionSystem.from_string('\n'.join(['2 A -> B', 'B -> 2A']), substance_factory=Substance)
    assert not rs2.check_balance(strict=True)
    assert rs2.composition_balance_vectors() == ([], [])