from __future__ import annotations
import logging
import typing as t
from copy import deepcopy
from textwrap import dedent
from traitlets.traitlets import (
from traitlets.utils import warnings
from traitlets.utils.bunch import Bunch
from traitlets.utils.text import indent, wrap_paragraphs
from .loader import Config, DeferredConfig, LazyConfigValue, _is_section_key
@classmethod
def section_names(cls) -> list[str]:
    """return section names as a list"""
    return [c.__name__ for c in reversed(cls.__mro__) if issubclass(c, Configurable) and issubclass(cls, c)]