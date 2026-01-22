import operator
from numba.core.imputils import (lower_builtin, lower_getattr_generic,
from numba.core import typing, types, cgutils
from numba.core.extending import overload_method, overload, intrinsic
@overload(operator.getitem)
def getitem_literal_idx(tup, idx):
    """
    Overloads BaseTuple getitem to cover cases where constant
    inference and RewriteConstGetitems cannot replace it
    with a static_getitem.
    """
    if not (isinstance(tup, types.BaseTuple) and isinstance(idx, types.IntegerLiteral)):
        return None
    idx_val = idx.literal_value

    def getitem_literal_idx_impl(tup, idx):
        return tup[idx_val]
    return getitem_literal_idx_impl