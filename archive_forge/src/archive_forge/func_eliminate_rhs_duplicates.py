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
def eliminate_rhs_duplicates(expr_list_list, ref_node_sequence):
    """Replace rhs items by LetRefNodes if they appear more than once.
    Creates a sequence of LetRefNodes that set up the required temps
    and appends them to ref_node_sequence.  The input list is modified
    in-place.
    """
    seen_nodes = set()
    ref_nodes = {}

    def find_duplicates(node):
        if node.is_literal or node.is_name:
            return
        if node in seen_nodes:
            if node not in ref_nodes:
                ref_node = LetRefNode(node)
                ref_nodes[node] = ref_node
                ref_node_sequence.append(ref_node)
        else:
            seen_nodes.add(node)
            if node.is_sequence_constructor:
                for item in node.args:
                    find_duplicates(item)
    for expr_list in expr_list_list:
        rhs = expr_list[-1]
        find_duplicates(rhs)
    if not ref_nodes:
        return

    def substitute_nodes(node):
        if node in ref_nodes:
            return ref_nodes[node]
        elif node.is_sequence_constructor:
            node.args = list(map(substitute_nodes, node.args))
        return node
    for node in ref_nodes:
        if node.is_sequence_constructor:
            node.args = list(map(substitute_nodes, node.args))
    for expr_list in expr_list_list:
        expr_list[-1] = substitute_nodes(expr_list[-1])