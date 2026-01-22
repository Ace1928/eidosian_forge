import collections.abc
import operator
from collections import defaultdict, Counter
from functools import reduce
import itertools
from itertools import accumulate
from typing import Optional, List, Tuple as tTuple
import typing
from sympy.core.numbers import Integer
from sympy.core.relational import Equality
from sympy.functions.special.tensor_functions import KroneckerDelta
from sympy.core.basic import Basic
from sympy.core.containers import Tuple
from sympy.core.expr import Expr
from sympy.core.function import (Function, Lambda)
from sympy.core.mul import Mul
from sympy.core.singleton import S
from sympy.core.sorting import default_sort_key
from sympy.core.symbol import (Dummy, Symbol)
from sympy.matrices.common import MatrixCommon
from sympy.matrices.expressions.diagonal import diagonalize_vector
from sympy.matrices.expressions.matexpr import MatrixExpr
from sympy.matrices.expressions.special import ZeroMatrix
from sympy.tensor.array.arrayop import (permutedims, tensorcontraction, tensordiagonal, tensorproduct)
from sympy.tensor.array.dense_ndim_array import ImmutableDenseNDimArray
from sympy.tensor.array.ndim_array import NDimArray
from sympy.tensor.indexed import (Indexed, IndexedBase)
from sympy.matrices.expressions.matexpr import MatrixElement
from sympy.tensor.array.expressions.utils import _apply_recursively_over_nested_lists, _sort_contraction_indices, \
from sympy.combinatorics import Permutation
from sympy.combinatorics.permutations import _af_invert
from sympy.core.sympify import _sympify
def to_array_contraction(self):
    counter = 0
    diag_indices = defaultdict(list)
    count_index_freq = Counter()
    for arg_with_ind in self.args_with_ind:
        count_index_freq.update(Counter(arg_with_ind.indices))
    free_index_count = count_index_freq[None]
    inv_perm1 = []
    inv_perm2 = []
    done = set()
    counter4 = 0
    for arg_with_ind in self.args_with_ind:
        counter2 = 0
        for i in arg_with_ind.indices:
            if i is None:
                inv_perm1.append(counter4)
                counter2 += 1
                counter4 += 1
                continue
            if i >= 0:
                continue
            diag_indices[-1 - i].append(counter + counter2)
            if count_index_freq[i] == 1 and i not in done:
                inv_perm1.append(free_index_count - 1 - i)
                done.add(i)
            elif i not in done:
                inv_perm2.append(free_index_count - 1 - i)
                done.add(i)
            counter2 += 1
        arg_with_ind.indices = [i if i is not None and i >= 0 else None for i in arg_with_ind.indices]
        counter += len([i for i in arg_with_ind.indices if i is None or i < 0])
    inverse_permutation = inv_perm1 + inv_perm2
    permutation = _af_invert(inverse_permutation)
    diag_indices_filtered = [tuple(v) for v in diag_indices.values() if len(v) > 1]
    self.merge_scalars()
    self.refresh_indices()
    args = [arg.element for arg in self.args_with_ind]
    contraction_indices = self.get_contraction_indices()
    expr = _array_contraction(_array_tensor_product(*args), *contraction_indices)
    expr2 = _array_diagonal(expr, *diag_indices_filtered)
    if self._track_permutation is not None:
        permutation2 = _af_invert([j for i in self._track_permutation for j in i])
        expr2 = _permute_dims(expr2, permutation2)
    expr3 = _permute_dims(expr2, permutation)
    return expr3