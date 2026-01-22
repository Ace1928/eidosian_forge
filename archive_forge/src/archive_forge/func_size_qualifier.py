from __future__ import annotations
from abc import (
import sys
from textwrap import dedent
from typing import TYPE_CHECKING
from pandas._config import get_option
from pandas.io.formats import format as fmt
from pandas.io.formats.printing import pprint_thing
@property
def size_qualifier(self) -> str:
    size_qualifier = ''
    if self.memory_usage:
        if self.memory_usage != 'deep':
            if 'object' in self.dtype_counts or self.data.index._is_memory_usage_qualified():
                size_qualifier = '+'
    return size_qualifier