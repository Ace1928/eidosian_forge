from __future__ import absolute_import
from .Errors import CompileError, error
from . import ExprNodes
from .ExprNodes import IntNode, NameNode, AttributeNode
from . import Options
from .Code import UtilityCode, TempitaUtilityCode
from .UtilityCode import CythonUtilityCode
from . import Buffer
from . import PyrexTypes
from . import ModuleNode
def get_memoryview_flag(access, packing):
    if access == 'full' and packing in ('strided', 'follow'):
        return 'generic'
    elif access == 'full' and packing == 'contig':
        return 'generic_contiguous'
    elif access == 'ptr' and packing in ('strided', 'follow'):
        return 'indirect'
    elif access == 'ptr' and packing == 'contig':
        return 'indirect_contiguous'
    elif access == 'direct' and packing in ('strided', 'follow'):
        return 'strided'
    else:
        assert (access, packing) == ('direct', 'contig'), (access, packing)
        return 'contiguous'