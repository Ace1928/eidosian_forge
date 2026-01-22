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
def get_is_contig_func_name(contig_type, ndim):
    assert contig_type in ('C', 'F')
    return '__pyx_memviewslice_is_contig_%s%d' % (contig_type, ndim)