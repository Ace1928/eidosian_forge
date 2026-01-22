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
def unobserve_all(self, name: str | t.Any=All) -> None:
    """Remove trait change handlers of any type for the specified name.
        If name is not specified, removes all trait notifiers."""
    if name is All:
        self._trait_notifiers = {}
    else:
        try:
            del self._trait_notifiers[name]
        except KeyError:
            pass