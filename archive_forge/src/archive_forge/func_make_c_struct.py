import collections
import warnings
from functools import cached_property
from llvmlite import ir
from .abstract import DTypeSpec, IteratorType, MutableSequence, Number, Type
from .common import Buffer, Opaque, SimpleIteratorType
from numba.core.typeconv import Conversion
from numba.core import utils
from .misc import UnicodeType
from .containers import Bytes
import numpy as np
@classmethod
def make_c_struct(cls, name_types):
    """Construct a Record type from a list of (name:str, type:Types).
        The layout of the structure will follow C.

        Note: only scalar types are supported currently.
        """
    from numba.core.registry import cpu_target
    ctx = cpu_target.target_context
    offset = 0
    fields = []
    lltypes = []
    for k, ty in name_types:
        if not isinstance(ty, (Number, NestedArray)):
            msg = 'Only Number and NestedArray types are supported, found: {}. '
            raise TypeError(msg.format(ty))
        if isinstance(ty, NestedArray):
            datatype = ctx.data_model_manager[ty].as_storage_type()
        else:
            datatype = ctx.get_data_type(ty)
        lltypes.append(datatype)
        size = ctx.get_abi_sizeof(datatype)
        align = ctx.get_abi_alignment(datatype)
        misaligned = offset % align
        if misaligned:
            offset += align - misaligned
        fields.append((k, {'type': ty, 'offset': offset, 'alignment': align}))
        offset += size
    abi_size = ctx.get_abi_sizeof(ir.LiteralStructType(lltypes))
    return Record(fields, size=abi_size, aligned=True)