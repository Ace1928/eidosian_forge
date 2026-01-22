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
def p_backquote_expr(s):
    pos = s.position()
    s.next()
    args = [p_test(s)]
    while s.sy == ',':
        s.next()
        args.append(p_test(s))
    s.expect('`')
    if len(args) == 1:
        arg = args[0]
    else:
        arg = ExprNodes.TupleNode(pos, args=args)
    return ExprNodes.BackquoteNode(pos, arg=arg)