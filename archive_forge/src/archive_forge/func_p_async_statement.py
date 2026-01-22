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
def p_async_statement(s, ctx, decorators):
    if s.sy == 'def':
        if 'pxd' in ctx.level:
            s.error('def statement not allowed here')
        s.level = ctx.level
        return p_def_statement(s, decorators, is_async_def=True)
    elif decorators:
        s.error('Decorators can only be followed by functions or classes')
    elif s.sy == 'for':
        return p_for_statement(s, is_async=True)
    elif s.sy == 'with':
        s.next()
        return p_with_items(s, is_async=True)
    else:
        s.error("expected one of 'def', 'for', 'with' after 'async'")