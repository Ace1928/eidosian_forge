from __future__ import annotations
import re
import sys
import warnings
from functools import wraps, lru_cache
from itertools import count
from typing import TYPE_CHECKING, Generic, Iterator, NamedTuple, TypeVar, TypedDict, overload
def store_tag(self, tag: str, attrs: dict[str, str], left_index: int, right_index: int) -> str:
    """Store tag data and return a placeholder."""
    self.tag_data.append({'tag': tag, 'attrs': attrs, 'left_index': left_index, 'right_index': right_index})
    placeholder = TAG_PLACEHOLDER % str(self.tag_counter)
    self.tag_counter += 1
    return placeholder