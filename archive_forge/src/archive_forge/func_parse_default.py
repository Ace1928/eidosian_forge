from __future__ import absolute_import
import re
import sys
import os
import tokenize
from io import StringIO
from ._looper import looper
from .compat3 import bytes, unicode_, basestring_, next, is_unicode, coerce_text
def parse_default(tokens, name, context):
    first, pos = tokens[0]
    assert first.startswith('default ')
    first = first.split(None, 1)[1]
    parts = first.split('=', 1)
    if len(parts) == 1:
        raise TemplateError('Expression must be {{default var=value}}; no = found in %r' % first, position=pos, name=name)
    var = parts[0].strip()
    if ',' in var:
        raise TemplateError('{{default x, y = ...}} is not supported', position=pos, name=name)
    if not var_re.search(var):
        raise TemplateError('Not a valid variable name for {{default}}: %r' % var, position=pos, name=name)
    expr = parts[1].strip()
    return (('default', pos, var, expr), tokens[1:])