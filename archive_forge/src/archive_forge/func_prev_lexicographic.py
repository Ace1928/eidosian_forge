from itertools import combinations
from sympy.combinatorics.graycode import GrayCode
def prev_lexicographic(self):
    """
        Generates the previous lexicographically ordered subset.

        Examples
        ========

        >>> from sympy.combinatorics import Subset
        >>> a = Subset([], ['a', 'b', 'c', 'd'])
        >>> a.prev_lexicographic().subset
        ['d']
        >>> a = Subset(['c','d'], ['a', 'b', 'c', 'd'])
        >>> a.prev_lexicographic().subset
        ['c']

        See Also
        ========

        next_lexicographic
        """
    i = self.superset_size - 1
    indices = Subset.subset_indices(self.subset, self.superset)
    while i >= 0 and i not in indices:
        i = i - 1
    if i == 0 or i - 1 in indices:
        indices.remove(i)
    else:
        if i >= 0:
            indices.remove(i)
            indices.append(i - 1)
        indices.append(self.superset_size - 1)
    ret_set = []
    super_set = self.superset
    for i in indices:
        ret_set.append(super_set[i])
    return Subset(ret_set, super_set)