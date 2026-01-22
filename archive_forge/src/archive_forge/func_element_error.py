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
def element_error(self, obj: t.Any, element: t.Any, validator: t.Any, side: str='Values') -> None:
    e = side + f" of the '{self.name}' trait of {class_of(obj)} instance must be {validator.info()}, but a value of {repr_type(element)} was specified."
    raise TraitError(e)