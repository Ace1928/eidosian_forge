from itertools import chain
import pytest
from ..util.testing import requires
from ..util.parsing import parsing_library
from ..units import default_units, units_library, allclose
from ..chemistry import Substance, Reaction
from ..reactionsystem import ReactionSystem
def test_ReactionSystem__subset():
    r1 = Reaction({'NH3': 2}, {'N2': 1, 'H2': 3})
    r2 = Reaction({'N2H4': 1}, {'N2': 1, 'H2': 2})
    rs1 = ReactionSystem([r1, r2])
    rs2, rs3 = rs1.subset(lambda r: 'N2H4' in r.keys())
    assert len(rs1.rxns) == 2 and len(rs2.rxns) == 1
    assert rs2 == ReactionSystem([r2])
    assert rs3 == ReactionSystem([r1])