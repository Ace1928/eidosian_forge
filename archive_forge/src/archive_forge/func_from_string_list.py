from __future__ import annotations
import contextlib
import enum
import inspect
import os
import re
import sys
import types
import typing as t
from ast import literal_eval
from .utils.bunch import Bunch
from .utils.descriptions import add_article, class_of, describe, repr_type
from .utils.getargspec import getargspec
from .utils.importstring import import_item
from .utils.sentinel import Sentinel
from .utils.warnings import deprecated_method, should_warn, warn
from all trait attributes.
def from_string_list(self, s_list: list[str]) -> t.Any:
    """Return a dict from a list of config strings.

        This is where we parse CLI configuration.

        Each item should have the form ``"key=value"``.

        item parsing is done in :meth:`.item_from_string`.
        """
    if len(s_list) == 1 and s_list[0] == 'None' and self.allow_none:
        return None
    if len(s_list) == 1 and s_list[0].startswith('{') and s_list[0].endswith('}'):
        warn(f'--{self.name}={s_list[0]} for dict-traits is deprecated in traitlets 5.0. You can pass --{self.name} <key=value> ... multiple times to add items to a dict.', DeprecationWarning, stacklevel=2)
        return literal_eval(s_list[0])
    combined = {}
    for d in [self.item_from_string(s) for s in s_list]:
        combined.update(d)
    return combined