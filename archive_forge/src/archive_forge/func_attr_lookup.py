import __main__
import abc
import glob
import itertools
import keyword
import logging
import os
import re
import rlcompleter
import builtins
from enum import Enum
from typing import (
from . import inspection
from . import line as lineparts
from .line import LinePart
from .lazyre import LazyReCompile
from .simpleeval import safe_eval, evaluate_current_expression, EvaluationError
from .importcompletion import ModuleGatherer
def attr_lookup(self, obj: Any, expr: str, attr: str) -> Iterator[str]:
    """Second half of attr_matches."""
    words = self.list_attributes(obj)
    if inspection.hasattr_safe(obj, '__class__'):
        words.append('__class__')
        klass = inspection.getattr_safe(obj, '__class__')
        words = words + rlcompleter.get_class_members(klass)
        if not isinstance(klass, abc.ABCMeta):
            try:
                words.remove('__abstractmethods__')
            except ValueError:
                pass
    n = len(attr)
    return (f'{expr}.{word}' for word in words if self.method_match(word, n, attr) and word != '__builtins__')