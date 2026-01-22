from __future__ import absolute_import
import cython
from . import Builtin
from . import ExprNodes
from . import Nodes
from . import Options
from . import PyrexTypes
from .Visitor import TreeVisitor, CythonTransform
from .Errors import error, warning, InternalError
def is_tracked(self, entry):
    if entry.is_anonymous:
        return False
    return entry.is_local or entry.is_pyclass_attr or entry.is_arg or entry.from_closure or entry.in_closure or entry.error_on_uninitialized