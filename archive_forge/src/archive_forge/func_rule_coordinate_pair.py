from __future__ import absolute_import
import re
from decimal import Decimal, getcontext
from functools import partial
def rule_coordinate_pair(self, next_val_fn, token):
    if token[0] not in self.number_tokens:
        raise SyntaxError('expecting a number; got %r' % (token,))
    x = getcontext().create_decimal(token[1])
    token = next_val_fn()
    if token[0] not in self.number_tokens:
        raise SyntaxError('expecting a number; got %r' % (token,))
    y = getcontext().create_decimal(token[1])
    token = next_val_fn()
    return ([x, y], token)