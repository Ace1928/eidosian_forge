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
def class_init(self, cls: type[t.Any], name: str | None) -> None:
    if isinstance(self._value_trait, TraitType):
        self._value_trait.class_init(cls, None)
    if isinstance(self._key_trait, TraitType):
        self._key_trait.class_init(cls, None)
    if self._per_key_traits is not None:
        for trait in self._per_key_traits.values():
            trait.class_init(cls, None)
    super().class_init(cls, name)