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
def class_print_help(cls, inst: HasTraits | None=None) -> None:
    """Get the help string for a single trait and print it."""
    print(cls.class_get_help(inst))