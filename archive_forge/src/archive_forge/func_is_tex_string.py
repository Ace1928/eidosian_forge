from __future__ import annotations
import logging # isort:skip
import re
from contextlib import contextmanager
from typing import (
from weakref import WeakKeyDictionary
from ..core.types import ID
from ..document.document import Document
from ..model import Model, collect_models
from ..settings import settings
from ..themes.theme import Theme
from ..util.dataclasses import dataclass, field
from ..util.serialization import (
def is_tex_string(text: str) -> bool:
    """ Whether a string begins and ends with MathJax default delimiters

    Args:
        text (str): String to check

    Returns:
        bool: True if string begins and ends with delimiters, False if not
    """
    dollars = '^\\$\\$.*?\\$\\$$'
    braces = '^\\\\\\[.*?\\\\\\]$'
    parens = '^\\\\\\(.*?\\\\\\)$'
    pat = re.compile(f'{dollars}|{braces}|{parens}', flags=re.S)
    return pat.match(text) is not None