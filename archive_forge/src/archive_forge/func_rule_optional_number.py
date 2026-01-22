from __future__ import absolute_import
import re
from decimal import Decimal
from functools import partial
from six.moves import range
def rule_optional_number(self, next_val_fn, token):
    if token[0] not in self.number_tokens:
        return (None, token)
    else:
        x = Decimal(token[1]) * 1
        token = next_val_fn()
        return (x, token)