from __future__ import annotations
import decimal
import re
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any, Callable
def newfangled_relation(self, left):
    if skip_token(self.tokens, 'symbol', '='):
        negated = False
    elif skip_token(self.tokens, 'symbol', '!='):
        negated = True
    else:
        raise RuleError('Expected "=" or "!=" or legacy relation')
    rv = ('relation', ('in', left, self.range_list()))
    return negate(rv) if negated else rv