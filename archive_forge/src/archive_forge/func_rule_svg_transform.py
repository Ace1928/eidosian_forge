from __future__ import absolute_import
import re
from decimal import Decimal
from functools import partial
from six.moves import range
def rule_svg_transform(self, next_val_fn, token):
    if token[0] != 'command':
        raise SyntaxError('expecting a transformation type; got %r' % (token,))
    command = token[1]
    rule = self.command_dispatch[command]
    token = next_val_fn()
    if token[0] != 'coordstart':
        raise SyntaxError("expecting '('; got %r" % (token,))
    numbers, token = rule(next_val_fn, token)
    if token[0] != 'coordend':
        raise SyntaxError("expecting ')'; got %r" % (token,))
    token = next_val_fn()
    return ((command, numbers), token)