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
def p_subscript_list(s):
    is_single_value = True
    items = [p_subscript(s)]
    while s.sy == ',':
        is_single_value = False
        s.next()
        if s.sy == ']':
            break
        items.append(p_subscript(s))
    return (items, is_single_value)