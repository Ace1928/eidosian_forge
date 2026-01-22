from itertools import combinations
from sympy.combinatorics.graycode import GrayCode
def next_lexicographic(self):
    """
        Generates the next lexicographically ordered subset.

        Examples
        ========

        >>> from sympy.combinatorics import Subset
        >>> a = Subset(['c', 'd'], ['a', 'b', 'c', 'd'])
        >>> a.next_lexicographic().subset
        ['d']
        >>> a = Subset(['d'], ['a', 'b', 'c', 'd'])
        >>> a.next_lexicographic().subset
        []

        See Also
        ========

        prev_lexicographic
        """
    i = self.superset_size - 1
    indices = Subset.subset_indices(self.subset, self.superset)
    if i in indices:
        if i - 1 in indices:
            indices.remove(i - 1)
        else:
            indices.remove(i)
            i = i - 1
            while i >= 0 and i not in indices:
                i = i - 1
            if i >= 0:
                indices.remove(i)
                indices.append(i + 1)
    else:
        while i not in indices and i >= 0:
            i = i - 1
        indices.append(i + 1)
    ret_set = []
    super_set = self.superset
    for i in indices:
        ret_set.append(super_set[i])
    return Subset(ret_set, super_set)