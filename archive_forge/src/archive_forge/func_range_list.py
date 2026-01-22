from __future__ import annotations
import decimal
import re
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any, Callable
def range_list(self):
    range_list = [self.range_or_value()]
    while skip_token(self.tokens, 'symbol', ','):
        range_list.append(self.range_or_value())
    return range_list_node(range_list)