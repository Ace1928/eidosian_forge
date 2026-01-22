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
def wrap_compile_time_constant(pos, value):
    rep = repr(value)
    if value is None:
        return ExprNodes.NoneNode(pos)
    elif value is Ellipsis:
        return ExprNodes.EllipsisNode(pos)
    elif isinstance(value, bool):
        return ExprNodes.BoolNode(pos, value=value)
    elif isinstance(value, int):
        return ExprNodes.IntNode(pos, value=rep, constant_result=value)
    elif isinstance(value, float):
        return ExprNodes.FloatNode(pos, value=rep, constant_result=value)
    elif isinstance(value, complex):
        node = ExprNodes.ImagNode(pos, value=repr(value.imag), constant_result=complex(0.0, value.imag))
        if value.real:
            node = ExprNodes.binop_node(pos, '+', ExprNodes.FloatNode(pos, value=repr(value.real), constant_result=value.real), node, constant_result=value)
        return node
    elif isinstance(value, _unicode):
        return ExprNodes.UnicodeNode(pos, value=EncodedString(value))
    elif isinstance(value, _bytes):
        bvalue = bytes_literal(value, 'ascii')
        return ExprNodes.BytesNode(pos, value=bvalue, constant_result=value)
    elif isinstance(value, tuple):
        args = [wrap_compile_time_constant(pos, arg) for arg in value]
        if None not in args:
            return ExprNodes.TupleNode(pos, args=args)
        else:
            return None
    elif not _IS_PY3 and isinstance(value, long):
        return ExprNodes.IntNode(pos, value=rep.rstrip('L'), constant_result=value)
    error(pos, 'Invalid type for compile-time constant: %r (type %s)' % (value, value.__class__.__name__))
    return None