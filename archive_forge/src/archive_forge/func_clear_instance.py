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
def clear_instance(cls) -> None:
    """unset _instance for this class and singleton parents."""
    if not cls.initialized():
        return
    for subclass in cls._walk_mro():
        if isinstance(subclass._instance, cls):
            subclass._instance = None