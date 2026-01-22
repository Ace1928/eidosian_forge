from sympy.combinatorics.permutations import Permutation, Perm
from sympy.combinatorics.tensor_can import (perm_af_direct_product, dummy_sgs,
from sympy.combinatorics.testutil import canonicalize_naive, graph_certificate
from sympy.testing.pytest import skip, XFAIL
@XFAIL
def test_riemann_invariants1():
    skip('takes too much time')
    baser, gensr = riemann_bsgs
    g = Permutation([17, 44, 11, 3, 0, 19, 23, 15, 38, 4, 25, 27, 43, 36, 22, 14, 8, 30, 41, 20, 2, 10, 12, 28, 18, 1, 29, 13, 37, 42, 33, 7, 9, 31, 24, 26, 39, 5, 34, 47, 32, 6, 21, 40, 35, 46, 45, 16, 48, 49])
    can = canonicalize(g, list(range(48)), 0, (baser, gensr, 12, 0))
    assert can == [0, 2, 4, 6, 1, 3, 8, 10, 5, 7, 12, 14, 9, 11, 16, 18, 13, 15, 20, 22, 17, 19, 24, 26, 21, 23, 28, 30, 25, 27, 32, 34, 29, 31, 36, 38, 33, 35, 40, 42, 37, 39, 44, 46, 41, 43, 45, 47, 48, 49]
    g = Permutation([0, 2, 4, 6, 7, 8, 10, 12, 14, 16, 18, 20, 19, 22, 24, 26, 5, 21, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 13, 48, 50, 52, 15, 49, 54, 56, 17, 33, 41, 58, 9, 23, 60, 62, 29, 35, 63, 64, 3, 45, 66, 68, 25, 37, 47, 57, 11, 31, 69, 70, 27, 39, 53, 72, 1, 59, 73, 74, 55, 61, 67, 76, 43, 65, 75, 78, 51, 71, 77, 79, 80, 81])
    can = canonicalize(g, list(range(80)), 0, (baser, gensr, 20, 0))
    assert can == [0, 2, 4, 6, 1, 8, 10, 12, 3, 14, 16, 18, 5, 20, 22, 24, 7, 26, 28, 30, 9, 15, 32, 34, 11, 36, 23, 38, 13, 40, 42, 44, 17, 39, 29, 46, 19, 48, 43, 50, 21, 45, 52, 54, 25, 56, 33, 58, 27, 60, 53, 62, 31, 51, 64, 66, 35, 65, 47, 68, 37, 70, 49, 72, 41, 74, 57, 76, 55, 67, 59, 78, 61, 69, 71, 75, 63, 79, 73, 77, 80, 81]