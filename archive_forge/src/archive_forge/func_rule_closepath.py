from __future__ import absolute_import
import re
from decimal import Decimal, getcontext
from functools import partial
def rule_closepath(self, next_val_fn, token):
    command = token[1]
    token = next_val_fn()
    return ((command, []), token)