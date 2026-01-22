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
def set_trait(self, name: str, value: t.Any) -> None:
    """Forcibly sets trait attribute, including read-only attributes."""
    cls = self.__class__
    if not self.has_trait(name):
        raise TraitError(f'Class {cls.__name__} does not have a trait named {name}')
    getattr(cls, name).set(self, value)