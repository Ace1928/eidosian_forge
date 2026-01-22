from sympy.core.basic import Basic
from sympy.core.containers import (Dict, Tuple)
from sympy.core.singleton import S
from sympy.core.sympify import _sympify
from sympy.tensor.array.mutable_ndim_array import MutableNDimArray
from sympy.tensor.array.ndim_array import NDimArray, ImmutableNDimArray
from sympy.utilities.iterables import flatten
import functools
def tomatrix(self):
    """
        Converts MutableDenseNDimArray to Matrix. Can convert only 2-dim array, else will raise error.

        Examples
        ========

        >>> from sympy import MutableSparseNDimArray
        >>> a = MutableSparseNDimArray([1 for i in range(9)], (3, 3))
        >>> b = a.tomatrix()
        >>> b
        Matrix([
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]])
        """
    from sympy.matrices import SparseMatrix
    if self.rank() != 2:
        raise ValueError('Dimensions must be of size of 2')
    mat_sparse = {}
    for key, value in self._sparse_array.items():
        mat_sparse[self._get_tuple_index(key)] = value
    return SparseMatrix(self.shape[0], self.shape[1], mat_sparse)