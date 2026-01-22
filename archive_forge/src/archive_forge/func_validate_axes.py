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
def validate_axes(pos, axes):
    if len(axes) >= Options.buffer_max_dims:
        error(pos, 'More dimensions than the maximum number of buffer dimensions were used.')
        return False
    return True