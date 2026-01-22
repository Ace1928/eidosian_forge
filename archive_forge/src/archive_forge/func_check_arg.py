import operator
from enum import IntEnum
from llvmlite import ir
from numba.core.extending import (
from numba.core.imputils import iternext_impl
from numba.core import types, cgutils
from numba.core.types import (
from numba.core.imputils import impl_ret_borrowed, RefType
from numba.core.errors import TypingError
from numba.core import typing
from numba.typed.typedobjectutils import (_as_bytes, _cast, _nonoptional,
from numba.cpython import listobj
def check_arg(arg, name):
    if not (arg is None or arg in index_types or isinstance(arg, (types.Omitted, types.NoneType))):
        raise TypingError('{} argument for index must be an integer'.format(name))