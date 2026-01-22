from sympy.core.singleton import S
from sympy.combinatorics.fp_groups import (FpGroup, low_index_subgroups,
from sympy.combinatorics.free_groups import (free_group, FreeGroup)
from sympy.testing.pytest import slow
def test_fp_subgroup():

    def _test_subgroup(K, T, S):
        _gens = T(K.generators)
        assert all((elem in S for elem in _gens))
        assert T.is_injective()
        assert T.image().order() == S.order()
    F, x, y = free_group('x, y')
    f = FpGroup(F, [x ** 4, y ** 2, x * y * x ** (-1) * y])
    S = FpSubgroup(f, [x * y])
    assert (x * y) ** (-3) in S
    K, T = f.subgroup([x * y], homomorphism=True)
    assert T(K.generators) == [y * x ** (-1)]
    _test_subgroup(K, T, S)
    S = FpSubgroup(f, [x ** (-1) * y * x])
    assert x ** (-1) * y ** 4 * x in S
    assert x ** (-1) * y ** 4 * x ** 2 not in S
    K, T = f.subgroup([x ** (-1) * y * x], homomorphism=True)
    assert T(K.generators[0] ** 3) == y ** 3
    _test_subgroup(K, T, S)
    f = FpGroup(F, [x ** 3, y ** 5, (x * y) ** 2])
    H = [x * y, x ** (-1) * y ** (-1) * x * y * x]
    K, T = f.subgroup(H, homomorphism=True)
    S = FpSubgroup(f, H)
    _test_subgroup(K, T, S)