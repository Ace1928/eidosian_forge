from _ast import Add
from _ast import And
from _ast import AST
from _ast import BitAnd
from _ast import BitOr
from _ast import BitXor
from _ast import Div
from _ast import Eq
from _ast import FloorDiv
from _ast import Gt
from _ast import GtE
from _ast import If
from _ast import In
from _ast import Invert
from _ast import Is
from _ast import IsNot
from _ast import LShift
from _ast import Lt
from _ast import LtE
from _ast import Mod
from _ast import Mult
from _ast import Name
from _ast import Not
from _ast import NotEq
from _ast import NotIn
from _ast import Or
from _ast import PyCF_ONLY_AST
from _ast import RShift
from _ast import Sub
from _ast import UAdd
from _ast import USub
def get_visitor(self, node):
    """
        Return the visitor function for this node or `None` if no visitor
        exists for this node.  In that case the generic visit function is
        used instead.
        """
    method = 'visit_' + node.__class__.__name__
    return getattr(self, method, None)