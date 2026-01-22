from __future__ import absolute_import
from .PyrexTypes import CType, CTypedefType, CStructOrUnionType
import cython
def pythran_indexing_code(indices):
    return _index_access(_index_code, indices)