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
def p_except_clause(s):
    pos = s.position()
    s.next()
    exc_type = None
    exc_value = None
    is_except_as = False
    if s.sy != ':':
        exc_type = p_test(s)
        if isinstance(exc_type, ExprNodes.TupleNode):
            exc_type = exc_type.args
        else:
            exc_type = [exc_type]
        if s.sy == ',' or (s.sy == 'IDENT' and s.systring == 'as' and (s.context.language_level == 2)):
            s.next()
            exc_value = p_test(s)
        elif s.sy == 'IDENT' and s.systring == 'as':
            s.next()
            pos2 = s.position()
            name = p_ident(s)
            exc_value = ExprNodes.NameNode(pos2, name=name)
            is_except_as = True
    body = p_suite(s)
    return Nodes.ExceptClauseNode(pos, pattern=exc_type, target=exc_value, body=body, is_except_as=is_except_as)