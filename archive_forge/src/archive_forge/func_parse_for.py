from __future__ import absolute_import
import re
import sys
import os
import tokenize
from io import StringIO
from ._looper import looper
from .compat3 import bytes, unicode_, basestring_, next, is_unicode, coerce_text
def parse_for(tokens, name, context):
    first, pos = tokens[0]
    tokens = tokens[1:]
    context = ('for',) + context
    content = []
    assert first.startswith('for '), first
    if first.endswith(':'):
        first = first[:-1]
    first = first[3:].strip()
    match = in_re.search(first)
    if not match:
        raise TemplateError('Bad for (no "in") in %r' % first, position=pos, name=name)
    vars = first[:match.start()]
    if '(' in vars:
        raise TemplateError('You cannot have () in the variable section of a for loop (%r)' % vars, position=pos, name=name)
    vars = tuple([v.strip() for v in first[:match.start()].split(',') if v.strip()])
    expr = first[match.end():]
    while 1:
        if not tokens:
            raise TemplateError('No {{endfor}}', position=pos, name=name)
        if isinstance(tokens[0], tuple) and tokens[0][0] == 'endfor':
            return (('for', pos, vars, expr, content), tokens[1:])
        next_chunk, tokens = parse_expr(tokens, name, context)
        content.append(next_chunk)