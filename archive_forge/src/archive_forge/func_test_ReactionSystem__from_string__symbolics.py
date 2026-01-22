from itertools import chain
import pytest
from ..util.testing import requires
from ..util.parsing import parsing_library
from ..units import default_units, units_library, allclose
from ..chemistry import Substance, Reaction
from ..reactionsystem import ReactionSystem
@requires(parsing_library, 'numpy')
def test_ReactionSystem__from_string__symbolics():
    rs3 = ReactionSystem.from_string("\nA -> B; 'kA'\nB -> C; 0\n", substance_factory=Substance)
    rs3.rxns[1].param = 2 * rs3.rxns[0].param
    assert rs3.rates(dict(A=29, B=31, kA=42)) == {'A': -29 * 42, 'B': 29 * 42 - 2 * 31 * 42, 'C': 2 * 31 * 42}