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
def p_c_class_options(s):
    objstruct_name = None
    typeobj_name = None
    check_size = None
    s.expect('[')
    while 1:
        if s.sy != 'IDENT':
            break
        if s.systring == 'object':
            s.next()
            objstruct_name = p_ident(s)
        elif s.systring == 'type':
            s.next()
            typeobj_name = p_ident(s)
        elif s.systring == 'check_size':
            s.next()
            check_size = p_ident(s)
            if check_size not in ('ignore', 'warn', 'error'):
                s.error('Expected one of ignore, warn or error, found %r' % check_size)
        if s.sy != ',':
            break
        s.next()
    s.expect(']', "Expected 'object', 'type' or 'check_size'")
    return (objstruct_name, typeobj_name, check_size)