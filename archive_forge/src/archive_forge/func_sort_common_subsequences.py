from __future__ import absolute_import
import cython
import copy
import hashlib
import sys
from . import PyrexTypes
from . import Naming
from . import ExprNodes
from . import Nodes
from . import Options
from . import Builtin
from . import Errors
from .Visitor import VisitorTransform, TreeVisitor
from .Visitor import CythonTransform, EnvTransform, ScopeTrackingTransform
from .UtilNodes import LetNode, LetRefNode
from .TreeFragment import TreeFragment
from .StringEncoding import EncodedString, _unicode
from .Errors import error, warning, CompileError, InternalError
from .Code import UtilityCode
def sort_common_subsequences(items):
    """Sort items/subsequences so that all items and subsequences that
    an item contains appear before the item itself.  This is needed
    because each rhs item must only be evaluated once, so its value
    must be evaluated first and then reused when packing sequences
    that contain it.

    This implies a partial order, and the sort must be stable to
    preserve the original order as much as possible, so we use a
    simple insertion sort (which is very fast for short sequences, the
    normal case in practice).
    """

    def contains(seq, x):
        for item in seq:
            if item is x:
                return True
            elif item.is_sequence_constructor and contains(item.args, x):
                return True
        return False

    def lower_than(a, b):
        return b.is_sequence_constructor and contains(b.args, a)
    for pos, item in enumerate(items):
        key = item[1]
        new_pos = pos
        for i in range(pos - 1, -1, -1):
            if lower_than(key, items[i][0]):
                new_pos = i
        if new_pos != pos:
            for i in range(pos, new_pos, -1):
                items[i] = items[i - 1]
            items[new_pos] = item