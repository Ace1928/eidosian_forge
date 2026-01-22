from sympy.core import Basic
from sympy.core.containers import Tuple
from sympy.tensor.array import Array
from sympy.core.sympify import _sympify
from sympy.utilities.iterables import flatten, iterable
from sympy.utilities.misc import as_int
from collections import defaultdict
@property
def tree_repr(self):
    """Returns the tree representation of the Prufer object.

        Examples
        ========

        >>> from sympy.combinatorics.prufer import Prufer
        >>> Prufer([[0, 3], [1, 3], [2, 3], [3, 4], [4, 5]]).tree_repr
        [[0, 3], [1, 3], [2, 3], [3, 4], [4, 5]]
        >>> Prufer([1, 0, 0]).tree_repr
        [[1, 2], [0, 1], [0, 3], [0, 4]]

        See Also
        ========

        to_tree

        """
    if self._tree_repr is None:
        self._tree_repr = self.to_tree(self._prufer_repr[:])
    return self._tree_repr