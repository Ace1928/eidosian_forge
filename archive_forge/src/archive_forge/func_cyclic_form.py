from sympy.combinatorics import Permutation as Perm
from sympy.combinatorics.perm_groups import PermutationGroup
from sympy.core import Basic, Tuple, default_sort_key
from sympy.sets import FiniteSet
from sympy.utilities.iterables import (minlex, unflatten, flatten)
from sympy.utilities.misc import as_int
@property
def cyclic_form(self):
    """Return the indices of the corners in cyclic notation.

        The indices are given relative to the original position of corners.

        See Also
        ========

        corners, array_form
        """
    return Perm._af_new(self.array_form).cyclic_form