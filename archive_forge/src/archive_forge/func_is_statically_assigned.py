from __future__ import absolute_import
import cython
from . import Builtin
from . import ExprNodes
from . import Nodes
from . import Options
from . import PyrexTypes
from .Visitor import TreeVisitor, CythonTransform
from .Errors import error, warning, InternalError
def is_statically_assigned(self, entry):
    if entry.is_local and entry.is_variable and (entry.type.is_struct_or_union or entry.type.is_complex or entry.type.is_array or (entry.type.is_cpp_class and (not entry.is_cpp_optional))):
        return True
    return False