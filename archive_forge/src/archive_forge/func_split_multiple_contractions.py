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
def split_multiple_contractions(self):
    """
        Recognize multiple contractions and attempt at rewriting them as paired-contractions.

        This allows some contractions involving more than two indices to be
        rewritten as multiple contractions involving two indices, thus allowing
        the expression to be rewritten as a matrix multiplication line.

        Examples:

        * `A_ij b_j0 C_jk` ===> `A*DiagMatrix(b)*C`

        Care for:
        - matrix being diagonalized (i.e. `A_ii`)
        - vectors being diagonalized (i.e. `a_i0`)

        Multiple contractions can be split into matrix multiplications if
        not more than two arguments are non-diagonals or non-vectors.
        Vectors get diagonalized while diagonal matrices remain diagonal.
        The non-diagonal matrices can be at the beginning or at the end
        of the final matrix multiplication line.
        """
    editor = _EditArrayContraction(self)
    contraction_indices = self.contraction_indices
    onearray_insert = []
    for indl, links in enumerate(contraction_indices):
        if len(links) <= 2:
            continue
        positions = editor.get_mapping_for_index(indl)
        current_dimension = self.expr.shape[links[0]]
        not_vectors = []
        vectors = []
        for arg_ind, rel_ind in positions:
            arg = editor.args_with_ind[arg_ind]
            mat = arg.element
            abs_arg_start, abs_arg_end = editor.get_absolute_range(arg)
            other_arg_pos = 1 - rel_ind
            other_arg_abs = abs_arg_start + other_arg_pos
            if 1 not in mat.shape or ((current_dimension == 1) is True and mat.shape != (1, 1)) or any((other_arg_abs in l for li, l in enumerate(contraction_indices) if li != indl)):
                not_vectors.append((arg, rel_ind))
            else:
                vectors.append((arg, rel_ind))
        if len(not_vectors) > 2:
            continue
        for v, rel_ind in vectors:
            v.element = diagonalize_vector(v.element)
        vectors_to_loop = not_vectors[:1] + vectors + not_vectors[1:]
        first_not_vector, rel_ind = vectors_to_loop[0]
        new_index = first_not_vector.indices[rel_ind]
        for v, rel_ind in vectors_to_loop[1:-1]:
            v.indices[rel_ind] = new_index
            new_index = editor.get_new_contraction_index()
            assert v.indices.index(None) == 1 - rel_ind
            v.indices[v.indices.index(None)] = new_index
            onearray_insert.append(v)
        last_vec, rel_ind = vectors_to_loop[-1]
        last_vec.indices[rel_ind] = new_index
    for v in onearray_insert:
        editor.insert_after(v, _ArgE(OneArray(1), [None]))
    return editor.to_array_contraction()