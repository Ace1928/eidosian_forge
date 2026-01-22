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
def p_exec_statement(s):
    pos = s.position()
    s.next()
    code = p_bit_expr(s)
    if isinstance(code, ExprNodes.TupleNode):
        tuple_variant = True
        args = code.args
        if len(args) not in (2, 3):
            s.error('expected tuple of length 2 or 3, got length %d' % len(args), pos=pos, fatal=False)
            args = [code]
    else:
        tuple_variant = False
        args = [code]
    if s.sy == 'in':
        if tuple_variant:
            s.error("tuple variant of exec does not support additional 'in' arguments", fatal=False)
        s.next()
        args.append(p_test(s))
        if s.sy == ',':
            s.next()
            args.append(p_test(s))
    return Nodes.ExecStatNode(pos, args=args)