from __future__ import annotations
import ast
import re
import typing as t
from dataclasses import dataclass
from string import Template
from types import CodeType
from urllib.parse import quote
from ..datastructures import iter_multi_items
from ..urls import _urlencode
from .converters import ValidationError
def build_compare_key(self) -> tuple[int, int, int]:
    """The build compare key for sorting.

        :internal:
        """
    return (1 if self.alias else 0, -len(self.arguments), -len(self.defaults or ()))