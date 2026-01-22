from __future__ import absolute_import
import re
import sys
import os
import tokenize
from io import StringIO
from ._looper import looper
from .compat3 import bytes, unicode_, basestring_, next, is_unicode, coerce_text
def parse_one_cond(tokens, name, context):
    (first, pos), tokens = (tokens[0], tokens[1:])
    content = []
    if first.endswith(':'):
        first = first[:-1]
    if first.startswith('if '):
        part = ('if', pos, first[3:].lstrip(), content)
    elif first.startswith('elif '):
        part = ('elif', pos, first[5:].lstrip(), content)
    elif first == 'else':
        part = ('else', pos, None, content)
    else:
        assert 0, 'Unexpected token %r at %s' % (first, pos)
    while 1:
        if not tokens:
            raise TemplateError('No {{endif}}', position=pos, name=name)
        if isinstance(tokens[0], tuple) and (tokens[0][0] == 'endif' or tokens[0][0].startswith('elif ') or tokens[0][0] == 'else'):
            return (part, tokens)
        next_chunk, tokens = parse_expr(tokens, name, context)
        content.append(next_chunk)