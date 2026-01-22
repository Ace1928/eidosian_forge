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
def looking_at_expr(s):
    if s.systring in base_type_start_words:
        return False
    elif s.sy == 'IDENT':
        is_type = False
        name = s.systring
        name_pos = s.position()
        dotted_path = []
        s.next()
        while s.sy == '.':
            s.next()
            dotted_path.append((s.systring, s.position()))
            s.expect('IDENT')
        saved = (s.sy, s.systring, s.position())
        if s.sy == 'IDENT':
            is_type = True
        elif s.sy == '*' or s.sy == '**':
            s.next()
            is_type = s.sy in (')', ']')
            s.put_back(*saved)
        elif s.sy == '(':
            s.next()
            is_type = s.sy == '*'
            s.put_back(*saved)
        elif s.sy == '[':
            s.next()
            is_type = s.sy == ']' or not looking_at_expr(s)
            s.put_back(*saved)
        dotted_path.reverse()
        for p in dotted_path:
            s.put_back(u'IDENT', *p)
            s.put_back(u'.', u'.', p[1])
        s.put_back(u'IDENT', name, name_pos)
        return not is_type and saved[0]
    else:
        return True