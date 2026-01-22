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
def p_simple_statement_list(s, ctx, first_statement=0):
    stat = p_simple_statement(s, first_statement=first_statement)
    pos = stat.pos
    stats = []
    if not isinstance(stat, Nodes.PassStatNode):
        stats.append(stat)
    while s.sy == ';':
        s.next()
        if s.sy in ('NEWLINE', 'EOF'):
            break
        stat = p_simple_statement(s, first_statement=first_statement)
        if isinstance(stat, Nodes.PassStatNode):
            continue
        stats.append(stat)
        first_statement = False
    if not stats:
        stat = Nodes.PassStatNode(pos)
    elif len(stats) == 1:
        stat = stats[0]
    else:
        stat = Nodes.StatListNode(pos, stats=stats)
    if s.sy not in ('NEWLINE', 'EOF'):
        if isinstance(stat, Nodes.ExprStatNode):
            if stat.expr.is_name and stat.expr.name == 'cdef':
                s.error("The 'cdef' keyword is only allowed in Cython files (pyx/pxi/pxd)", pos)
    s.expect_newline('Syntax error in simple statement list')
    return stat