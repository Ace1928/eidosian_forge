from __future__ import annotations
import decimal
import re
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any, Callable
def compile_not(self, relation):
    return self.compile_relation(*relation[1], negated=True)