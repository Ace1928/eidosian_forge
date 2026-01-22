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
def p_int_literal(s):
    pos = s.position()
    value = s.systring
    s.next()
    unsigned = ''
    longness = ''
    while value[-1] in u'UuLl':
        if value[-1] in u'Ll':
            longness += 'L'
        else:
            unsigned += 'U'
        value = value[:-1]
    is_c_literal = None
    if unsigned:
        is_c_literal = True
    elif longness:
        if longness == 'LL' or s.context.language_level >= 3:
            is_c_literal = True
    if s.in_python_file:
        if is_c_literal:
            error(pos, 'illegal integer literal syntax in Python source file')
        is_c_literal = False
    return ExprNodes.IntNode(pos, is_c_literal=is_c_literal, value=value, unsigned=unsigned, longness=longness)