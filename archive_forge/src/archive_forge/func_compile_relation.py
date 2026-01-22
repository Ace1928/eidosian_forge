from __future__ import annotations
import decimal
import re
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any, Callable
def compile_relation(self, method, expr, range_list, negated=False):
    ranges = []
    for item in range_list[1]:
        if item[0] == item[1]:
            ranges.append(self.compile(item[0]))
        else:
            ranges.append(f'{self.compile(item[0])}..{self.compile(item[1])}')
    return f'{self.compile(expr)}{(' not' if negated else '')} {method} {','.join(ranges)}'