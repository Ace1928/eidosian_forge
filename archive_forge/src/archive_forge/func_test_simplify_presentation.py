from sympy.core.singleton import S
from sympy.combinatorics.fp_groups import (FpGroup, low_index_subgroups,
from sympy.combinatorics.free_groups import (free_group, FreeGroup)
from sympy.testing.pytest import slow
def test_simplify_presentation():
    G = simplify_presentation(FpGroup(FreeGroup([]), []))
    assert not G.generators
    assert not G.relators