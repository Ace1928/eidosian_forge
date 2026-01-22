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
def p_cat_string_literal(s):
    pos = s.position()
    kind, bytes_value, unicode_value = p_string_literal(s)
    if kind == 'c' or s.sy != 'BEGIN_STRING':
        return (kind, bytes_value, unicode_value)
    bstrings, ustrings, positions = ([bytes_value], [unicode_value], [pos])
    bytes_value = unicode_value = None
    while s.sy == 'BEGIN_STRING':
        pos = s.position()
        next_kind, next_bytes_value, next_unicode_value = p_string_literal(s)
        if next_kind == 'c':
            error(pos, 'Cannot concatenate char literal with another string or char literal')
            continue
        elif next_kind != kind:
            if {kind, next_kind} in ({'f', 'u'}, {'f', ''}):
                kind = 'f'
            else:
                error(pos, "Cannot mix string literals of different types, expected %s'', got %s''" % (kind, next_kind))
                continue
        bstrings.append(next_bytes_value)
        ustrings.append(next_unicode_value)
        positions.append(pos)
    if kind in ('b', 'c', '') or (kind == 'u' and None not in bstrings):
        bytes_value = bytes_literal(StringEncoding.join_bytes(bstrings), s.source_encoding)
    if kind in ('u', ''):
        unicode_value = EncodedString(u''.join([u for u in ustrings if u is not None]))
    if kind == 'f':
        unicode_value = []
        for u, pos in zip(ustrings, positions):
            if isinstance(u, list):
                unicode_value += u
            else:
                unicode_value.append(ExprNodes.UnicodeNode(pos, value=EncodedString(u)))
    return (kind, bytes_value, unicode_value)