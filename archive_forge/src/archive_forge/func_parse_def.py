from __future__ import absolute_import
import re
import sys
import os
import tokenize
from io import StringIO
from ._looper import looper
from .compat3 import bytes, unicode_, basestring_, next, is_unicode, coerce_text
def parse_def(tokens, name, context):
    first, start = tokens[0]
    tokens = tokens[1:]
    assert first.startswith('def ')
    first = first.split(None, 1)[1]
    if first.endswith(':'):
        first = first[:-1]
    if '(' not in first:
        func_name = first
        sig = ((), None, None, {})
    elif not first.endswith(')'):
        raise TemplateError("Function definition doesn't end with ): %s" % first, position=start, name=name)
    else:
        first = first[:-1]
        func_name, sig_text = first.split('(', 1)
        sig = parse_signature(sig_text, name, start)
    context = context + ('def',)
    content = []
    while 1:
        if not tokens:
            raise TemplateError('Missing {{enddef}}', position=start, name=name)
        if isinstance(tokens[0], tuple) and tokens[0][0] == 'enddef':
            return (('def', start, func_name, sig, content), tokens[1:])
        next_chunk, tokens = parse_expr(tokens, name, context)
        content.append(next_chunk)