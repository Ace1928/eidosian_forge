from __future__ import absolute_import
import cython
from io import StringIO
import re
import sys
from unicodedata import lookup as lookup_unicodechar, category as unicode_category
from functools import partial, reduce
from .Scanning import PyrexScanner, FileSourceDescriptor, tentatively_scan
from . import Nodes
from . import ExprNodes
from . import Builtin
from . import StringEncoding
from .StringEncoding import EncodedString, bytes_literal, _unicode, _bytes
from .ModuleNode import ModuleNode
from .Errors import error, warning
from .. import Utils
from . import Future
from . import Options
def p_expression_or_assignment(s):
    expr = p_testlist_star_expr(s)
    has_annotation = False
    if s.sy == ':' and (expr.is_name or expr.is_subscript or expr.is_attribute):
        has_annotation = True
        s.next()
        expr.annotation = p_annotation(s)
    if s.sy == '=' and expr.is_starred:
        s.error('a starred assignment target must be in a list or tuple - maybe you meant to use an index assignment: var[0] = ...', pos=expr.pos)
    expr_list = [expr]
    while s.sy == '=':
        s.next()
        if s.sy == 'yield':
            expr = p_yield_expression(s)
        else:
            expr = p_testlist_star_expr(s)
        expr_list.append(expr)
    if len(expr_list) == 1:
        if re.match('([-+*/%^&|]|<<|>>|\\*\\*|//|@)=', s.sy):
            lhs = expr_list[0]
            if isinstance(lhs, ExprNodes.SliceIndexNode):
                lhs = ExprNodes.IndexNode(lhs.pos, base=lhs.base, index=make_slice_node(lhs.pos, lhs.start, lhs.stop))
            elif not isinstance(lhs, (ExprNodes.AttributeNode, ExprNodes.IndexNode, ExprNodes.NameNode)):
                error(lhs.pos, 'Illegal operand for inplace operation.')
            operator = s.sy[:-1]
            s.next()
            if s.sy == 'yield':
                rhs = p_yield_expression(s)
            else:
                rhs = p_testlist(s)
            return Nodes.InPlaceAssignmentNode(lhs.pos, operator=operator, lhs=lhs, rhs=rhs)
        expr = expr_list[0]
        return Nodes.ExprStatNode(expr.pos, expr=expr)
    rhs = expr_list[-1]
    if len(expr_list) == 2:
        return Nodes.SingleAssignmentNode(rhs.pos, lhs=expr_list[0], rhs=rhs, first=has_annotation)
    else:
        return Nodes.CascadedAssignmentNode(rhs.pos, lhs_list=expr_list[:-1], rhs=rhs)