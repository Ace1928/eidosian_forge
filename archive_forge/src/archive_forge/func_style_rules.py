from __future__ import annotations
import itertools
import re
from enum import Enum
from typing import Hashable, TypeVar
from prompt_toolkit.cache import SimpleCache
from .base import (
from .named_colors import NAMED_COLORS
@property
def style_rules(self) -> list[tuple[str, str]]:
    style_rules = []
    for s in self.styles:
        style_rules.extend(s.style_rules)
    return style_rules