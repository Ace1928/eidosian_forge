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
def setup_instance(*args: t.Any, **kwargs: t.Any) -> None:
    self = args[0]
    args = args[1:]
    self._trait_values = self._static_immutable_initial_values.copy()
    self._trait_notifiers = {}
    self._trait_validators = {}
    self._cross_validation_lock = False
    super(HasTraits, self).setup_instance(*args, **kwargs)