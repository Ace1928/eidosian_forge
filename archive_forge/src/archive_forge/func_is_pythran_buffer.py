from __future__ import absolute_import
from .PyrexTypes import CType, CTypedefType, CStructOrUnionType
import cython
def is_pythran_buffer(type_):
    return type_.is_numpy_buffer and is_pythran_supported_dtype(type_.dtype) and (type_.mode in ('c', 'strided')) and (not type_.cast)