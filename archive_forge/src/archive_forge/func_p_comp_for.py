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
def p_comp_for(s, body):
    pos = s.position()
    is_async = False
    if s.sy == 'async':
        is_async = True
        s.next()
    s.expect('for')
    kw = p_for_bounds(s, allow_testlist=False, is_async=is_async)
    kw.update(else_clause=None, body=p_comp_iter(s, body), is_async=is_async)
    return Nodes.ForStatNode(pos, **kw)