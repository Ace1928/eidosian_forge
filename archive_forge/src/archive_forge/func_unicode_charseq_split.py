import operator
import numpy as np
from llvmlite import ir
from numba.core import types, cgutils
from numba.core.extending import (overload, intrinsic, overload_method,
from numba.core.cgutils import is_nonelike
from numba.cpython import unicode
@overload_method(types.UnicodeCharSeq, 'split')
@overload_method(types.CharSeq, 'split')
@overload_method(types.Bytes, 'split')
def unicode_charseq_split(a, sep=None, maxsplit=-1):
    if not (maxsplit == -1 or isinstance(maxsplit, (types.Omitted, types.Integer, types.IntegerLiteral))):
        return None
    if isinstance(a, types.UnicodeCharSeq):
        if isinstance(sep, types.UnicodeCharSeq):

            def impl(a, sep=None, maxsplit=-1):
                return str(a).split(sep=str(sep), maxsplit=maxsplit)
            return impl
        if isinstance(sep, types.UnicodeType):

            def impl(a, sep=None, maxsplit=-1):
                return str(a).split(sep=sep, maxsplit=maxsplit)
            return impl
        if is_nonelike(sep):
            if is_default(maxsplit, -1):

                def impl(a, sep=None, maxsplit=-1):
                    return str(a).split()
            else:

                def impl(a, sep=None, maxsplit=-1):
                    return str(a).split(maxsplit=maxsplit)
            return impl
    if isinstance(a, (types.CharSeq, types.Bytes)):
        if isinstance(sep, (types.CharSeq, types.Bytes)):

            def impl(a, sep=None, maxsplit=-1):
                return _map_bytes(a._to_str().split(sep._to_str(), maxsplit=maxsplit))
            return impl
        if is_nonelike(sep):
            if is_default(maxsplit, -1):

                def impl(a, sep=None, maxsplit=-1):
                    return _map_bytes(a._to_str().split())
            else:

                def impl(a, sep=None, maxsplit=-1):
                    return _map_bytes(a._to_str().split(maxsplit=maxsplit))
            return impl