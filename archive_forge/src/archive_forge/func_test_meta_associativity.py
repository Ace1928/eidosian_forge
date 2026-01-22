import random
from .links_base import CrossingEntryPoint
from ..sage_helper import _within_sage
def test_meta_associativity():
    """
    Tests strand_matrix_merge for required invariance properties.
    """

    def eval_merges(merges):
        R = PolynomialRing(QQ, 'x', 49).fraction_field()
        A = matrix(7, 7, R.gens())
        for a, b in merges:
            A = strand_matrix_merge(A, a, b)
        return A
    associative_merges = [([(0, 1), (0, 1)], [(1, 2), (0, 1)]), ([(0, 1), (0, 2)], [(1, 3), (0, 1)]), ([(2, 5), (2, 3)], [(5, 3), (2, 3)])]
    for m1, m2 in associative_merges:
        assert eval_merges(m1) == eval_merges(m2)