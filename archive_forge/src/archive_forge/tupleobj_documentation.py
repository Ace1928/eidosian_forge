import operator
from numba.core.imputils import (lower_builtin, lower_getattr_generic,
from numba.core import typing, types, cgutils
from numba.core.extending import overload_method, overload, intrinsic

    Overloads BaseTuple getitem to cover cases where constant
    inference and RewriteConstGetitems cannot replace it
    with a static_getitem.
    