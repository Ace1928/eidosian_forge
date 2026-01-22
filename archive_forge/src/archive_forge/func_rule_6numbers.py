from __future__ import absolute_import
import re
from decimal import Decimal
from functools import partial
from six.moves import range
def rule_6numbers(self, next_val_fn, token):
    numbers = []
    token = next_val_fn()
    for i in range(6):
        number, token = self.rule_number(next_val_fn, token)
        numbers.append(number)
    return (numbers, token)