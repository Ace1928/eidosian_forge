from __future__ import absolute_import
from .PyrexTypes import CType, CTypedefType, CStructOrUnionType
import cython
def is_pythran_supported_node_or_none(node):
    return node.is_none or is_pythran_supported_type(node.type)