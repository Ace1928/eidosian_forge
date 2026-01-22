from itertools import combinations
from sympy.combinatorics.graycode import GrayCode
@property
def superset_size(self):
    """
        Returns the size of the superset.

        Examples
        ========

        >>> from sympy.combinatorics import Subset
        >>> a = Subset(['c', 'd'], ['a', 'b', 'c', 'd'])
        >>> a.superset_size
        4

        See Also
        ========

        subset, superset, size, cardinality
        """
    return len(self.superset)