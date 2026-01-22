from itertools import chain
import pytest
from ..util.testing import requires
from ..util.parsing import parsing_library
from ..units import default_units, units_library, allclose
from ..chemistry import Substance, Reaction
from ..reactionsystem import ReactionSystem
@requires('numpy')
def test_ReactionSystem__upper_conc_bounds():
    rs = ReactionSystem.from_string('\n'.join(['2 NH3 -> N2 + 3 H2', 'N2H4 -> N2 +   2  H2']))
    c0 = {'NH3': 5, 'N2': 7, 'H2': 11, 'N2H4': 2}
    _N = 5 + 14 + 4
    _H = 15 + 22 + 8
    ref = {'NH3': min(_N, _H / 3), 'N2': _N / 2, 'H2': _H / 2, 'N2H4': min(_N / 2, _H / 4)}
    res = rs.as_per_substance_dict(rs.upper_conc_bounds(c0))
    assert res == ref