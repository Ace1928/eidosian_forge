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
def put_init_entry(mv_cname, code):
    code.putln('%s.data = NULL;' % mv_cname)
    code.putln('%s.memview = NULL;' % mv_cname)