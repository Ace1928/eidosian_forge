import re
def triples_with_fixed_sum_iterator(N, skipVertices=False):
    """
    Similar to tuples_with_fixed_sum_iterator for triples.

    >>> list(triples_with_fixed_sum_iterator(2, skipVertices = True))
    [(0, 1, 1), (1, 0, 1), (1, 1, 0)]
    """
    return tuples_with_fixed_sum_iterator(N, 3, skipVertices=skipVertices)