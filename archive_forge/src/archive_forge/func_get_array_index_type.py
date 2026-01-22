import numpy as np
import operator
from collections import namedtuple
from numba.core import types, utils
from numba.core.typing.templates import (AttributeTemplate, AbstractTemplate,
from numba.core.typing import collections
from numba.core.errors import (TypingError, RequireLiteralValue, NumbaTypeError,
from numba.core.cgutils import is_nonelike
def get_array_index_type(ary, idx):
    """
    Returns None or a tuple-3 for the types of the input array, index, and
    resulting type of ``array[index]``.

    Note: This is shared logic for ndarray getitem and setitem.
    """
    if not isinstance(ary, types.Buffer):
        return
    ndim = ary.ndim
    left_indices = []
    right_indices = []
    ellipsis_met = False
    advanced = False
    num_newaxis = 0
    if not isinstance(idx, types.BaseTuple):
        idx = [idx]
    in_subspace = False
    num_subspaces = 0
    array_indices = 0
    for ty in idx:
        if ty is types.ellipsis:
            if ellipsis_met:
                raise NumbaTypeError('Only one ellipsis allowed in array indices (got %s)' % (idx,))
            ellipsis_met = True
            in_subspace = False
        elif isinstance(ty, types.SliceType):
            in_subspace = False
        elif isinstance(ty, types.Integer):
            ty = types.intp if ty.signed else types.uintp
            ndim -= 1
            if not in_subspace:
                num_subspaces += 1
                in_subspace = True
        elif isinstance(ty, types.Array) and ty.ndim == 0 and isinstance(ty.dtype, types.Integer):
            ndim -= 1
            if not in_subspace:
                num_subspaces += 1
                in_subspace = True
        elif isinstance(ty, types.Array) and isinstance(ty.dtype, (types.Integer, types.Boolean)):
            if ty.ndim > 1:
                raise NumbaTypeError('Multi-dimensional indices are not supported.')
            array_indices += 1
            advanced = True
            if not in_subspace:
                num_subspaces += 1
                in_subspace = True
        elif is_nonelike(ty):
            ndim += 1
            num_newaxis += 1
        else:
            raise NumbaTypeError('Unsupported array index type %s in %s' % (ty, idx))
        (right_indices if ellipsis_met else left_indices).append(ty)
    if advanced:
        if array_indices > 1:
            msg = 'Using more than one non-scalar array index is unsupported.'
            raise NumbaTypeError(msg)
        if num_subspaces > 1:
            msg = 'Using more than one indexing subspace is unsupported. An indexing subspace is a group of one or more consecutive indices comprising integer or array types.'
            raise NumbaTypeError(msg)
    if advanced and (not isinstance(ary, types.Array)):
        return
    all_indices = left_indices + right_indices
    if ellipsis_met:
        assert right_indices[0] is types.ellipsis
        del right_indices[0]
    n_indices = len(all_indices) - ellipsis_met - num_newaxis
    if n_indices > ary.ndim:
        raise NumbaTypeError('cannot index %s with %d indices: %s' % (ary, n_indices, idx))
    if n_indices == ary.ndim and ndim == 0 and (not ellipsis_met):
        res = ary.dtype
    elif advanced:
        res = ary.copy(ndim=ndim, layout='C', readonly=False)
    else:
        if ary.slice_is_copy:
            return
        layout = ary.layout

        def keeps_contiguity(ty, is_innermost):
            return ty is types.ellipsis or isinstance(ty, types.Integer) or (is_innermost and isinstance(ty, types.SliceType) and (not ty.has_step))

        def check_contiguity(outer_indices):
            """
            Whether indexing with the given indices (from outer to inner in
            physical layout order) can keep an array contiguous.
            """
            for ty in outer_indices[:-1]:
                if not keeps_contiguity(ty, False):
                    return False
            if outer_indices and (not keeps_contiguity(outer_indices[-1], True)):
                return False
            return True
        if layout == 'C':
            if n_indices == ary.ndim:
                left_indices = left_indices + right_indices
                right_indices = []
            if right_indices:
                layout = 'A'
            elif not check_contiguity(left_indices):
                layout = 'A'
        elif layout == 'F':
            if n_indices == ary.ndim:
                right_indices = left_indices + right_indices
                left_indices = []
            if left_indices:
                layout = 'A'
            elif not check_contiguity(right_indices[::-1]):
                layout = 'A'
        if ndim == 0:
            res = ary.dtype
        else:
            res = ary.copy(ndim=ndim, layout=layout)
    if isinstance(idx, types.BaseTuple):
        idx = types.BaseTuple.from_types(all_indices)
    else:
        idx, = all_indices
    return Indexing(idx, res, advanced)