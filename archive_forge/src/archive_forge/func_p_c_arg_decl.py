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
def p_c_arg_decl(s, ctx, in_pyfunc, cmethod_flag=0, nonempty=0, kw_only=0, annotated=1):
    pos = s.position()
    not_none = or_none = 0
    default = None
    annotation = None
    if s.in_python_file:
        base_type = Nodes.CSimpleBaseTypeNode(pos, name=None, module_path=[], is_basic_c_type=0, signed=0, complex=0, longness=0, is_self_arg=cmethod_flag, templates=None)
    else:
        base_type = p_c_base_type(s, nonempty=nonempty)
    declarator = p_c_declarator(s, ctx, nonempty=nonempty)
    if s.sy in ('not', 'or') and (not s.in_python_file):
        kind = s.sy
        s.next()
        if s.sy == 'IDENT' and s.systring == 'None':
            s.next()
        else:
            s.error("Expected 'None'")
        if not in_pyfunc:
            error(pos, "'%s None' only allowed in Python functions" % kind)
        or_none = kind == 'or'
        not_none = kind == 'not'
    if annotated and s.sy == ':':
        s.next()
        annotation = p_annotation(s)
    if s.sy == '=':
        s.next()
        if 'pxd' in ctx.level:
            if s.sy in ['*', '?']:
                default = ExprNodes.NoneNode(pos)
                s.next()
            elif 'inline' in ctx.modifiers:
                default = p_test(s)
            else:
                error(pos, 'default values cannot be specified in pxd files, use ? or *')
        else:
            default = p_test(s)
    return Nodes.CArgDeclNode(pos, base_type=base_type, declarator=declarator, not_none=not_none, or_none=or_none, default=default, annotation=annotation, kw_only=kw_only)