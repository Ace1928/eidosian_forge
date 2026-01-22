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
def p_namedexpr_test_or_starred_expr_list(s, expr=None):
    exprs = expr is not None and [expr] or []
    while s.sy not in expr_terminators:
        exprs.append(p_namedexpr_test_or_starred_expr(s))
        if s.sy != ',':
            break
        s.next()
    return exprs