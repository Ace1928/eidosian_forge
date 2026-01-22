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
def get_default_value(self) -> G | None:
    """DEPRECATED: Retrieve the static default value for this trait.
        Use self.default_value instead
        """
    warn('get_default_value is deprecated in traitlets 4.0: use the .default_value attribute', DeprecationWarning, stacklevel=2)
    return t.cast(G, self.default_value)