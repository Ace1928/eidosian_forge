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
def p_string_literal(s, kind_override=None):
    pos = s.position()
    is_python3_source = s.context.language_level >= 3
    has_non_ascii_literal_characters = False
    string_start_pos = (pos[0], pos[1], pos[2] + len(s.systring))
    kind_string = s.systring.rstrip('"\'').lower()
    if len(kind_string) > 1:
        if len(set(kind_string)) != len(kind_string):
            error(pos, 'Duplicate string prefix character')
        if 'b' in kind_string and 'u' in kind_string:
            error(pos, 'String prefixes b and u cannot be combined')
        if 'b' in kind_string and 'f' in kind_string:
            error(pos, 'String prefixes b and f cannot be combined')
        if 'u' in kind_string and 'f' in kind_string:
            error(pos, 'String prefixes u and f cannot be combined')
    is_raw = 'r' in kind_string
    if 'c' in kind_string:
        if len(kind_string) != 1:
            error(pos, 'Invalid string prefix for character literal')
        kind = 'c'
    elif 'f' in kind_string:
        kind = 'f'
        is_raw = True
    elif 'b' in kind_string:
        kind = 'b'
    elif 'u' in kind_string:
        kind = 'u'
    else:
        kind = ''
    if kind == '' and kind_override is None and (Future.unicode_literals in s.context.future_directives):
        chars = StringEncoding.StrLiteralBuilder(s.source_encoding)
        kind = 'u'
    else:
        if kind_override is not None and kind_override in 'ub':
            kind = kind_override
        if kind in ('u', 'f'):
            chars = StringEncoding.UnicodeLiteralBuilder()
        elif kind == '':
            chars = StringEncoding.StrLiteralBuilder(s.source_encoding)
        else:
            chars = StringEncoding.BytesLiteralBuilder(s.source_encoding)
    while 1:
        s.next()
        sy = s.sy
        systr = s.systring
        if sy == 'CHARS':
            chars.append(systr)
            if is_python3_source and (not has_non_ascii_literal_characters) and check_for_non_ascii_characters(systr):
                has_non_ascii_literal_characters = True
        elif sy == 'ESCAPE':
            if is_raw and (is_python3_source or kind != 'u' or systr[1] not in u'Uu'):
                chars.append(systr)
                if is_python3_source and (not has_non_ascii_literal_characters) and check_for_non_ascii_characters(systr):
                    has_non_ascii_literal_characters = True
            else:
                _append_escape_sequence(kind, chars, systr, s)
        elif sy == 'NEWLINE':
            chars.append(u'\n')
        elif sy == 'END_STRING':
            break
        elif sy == 'EOF':
            s.error('Unclosed string literal', pos=pos)
        else:
            s.error('Unexpected token %r:%r in string literal' % (sy, s.systring))
    if kind == 'c':
        unicode_value = None
        bytes_value = chars.getchar()
        if len(bytes_value) != 1:
            error(pos, u'invalid character literal: %r' % bytes_value)
    else:
        bytes_value, unicode_value = chars.getstrings()
        if has_non_ascii_literal_characters and is_python3_source and (Future.unicode_literals in s.context.future_directives):
            if kind == 'b':
                s.error('bytes can only contain ASCII literal characters.', pos=pos)
            bytes_value = None
    if kind == 'f':
        unicode_value = p_f_string(s, unicode_value, string_start_pos, is_raw='r' in kind_string)
    s.next()
    return (kind, bytes_value, unicode_value)