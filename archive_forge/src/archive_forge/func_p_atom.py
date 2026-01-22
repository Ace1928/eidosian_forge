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
def p_atom(s):
    pos = s.position()
    sy = s.sy
    if sy == '(':
        s.next()
        if s.sy == ')':
            result = ExprNodes.TupleNode(pos, args=[])
        elif s.sy == 'yield':
            result = p_yield_expression(s)
        else:
            result = p_testlist_comp(s)
        s.expect(')')
        return result
    elif sy == '[':
        return p_list_maker(s)
    elif sy == '{':
        return p_dict_or_set_maker(s)
    elif sy == '`':
        return p_backquote_expr(s)
    elif sy == '...':
        expect_ellipsis(s)
        return ExprNodes.EllipsisNode(pos)
    elif sy == 'INT':
        return p_int_literal(s)
    elif sy == 'FLOAT':
        value = s.systring
        s.next()
        return ExprNodes.FloatNode(pos, value=value)
    elif sy == 'IMAG':
        value = s.systring[:-1]
        s.next()
        return ExprNodes.ImagNode(pos, value=value)
    elif sy == 'BEGIN_STRING':
        kind, bytes_value, unicode_value = p_cat_string_literal(s)
        if kind == 'c':
            return ExprNodes.CharNode(pos, value=bytes_value)
        elif kind == 'u':
            return ExprNodes.UnicodeNode(pos, value=unicode_value, bytes_value=bytes_value)
        elif kind == 'b':
            return ExprNodes.BytesNode(pos, value=bytes_value)
        elif kind == 'f':
            return ExprNodes.JoinedStrNode(pos, values=unicode_value)
        elif kind == '':
            return ExprNodes.StringNode(pos, value=bytes_value, unicode_value=unicode_value)
        else:
            s.error("invalid string kind '%s'" % kind)
    elif sy == 'IDENT':
        name = s.systring
        if name == 'None':
            result = ExprNodes.NoneNode(pos)
        elif name == 'True':
            result = ExprNodes.BoolNode(pos, value=True)
        elif name == 'False':
            result = ExprNodes.BoolNode(pos, value=False)
        elif name == 'NULL' and (not s.in_python_file):
            result = ExprNodes.NullNode(pos)
        else:
            result = p_name(s, name)
        s.next()
        return result
    else:
        s.error('Expected an identifier or literal')