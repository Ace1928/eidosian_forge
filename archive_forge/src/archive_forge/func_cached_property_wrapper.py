import ast
import itertools
import types
from collections import OrderedDict, Counter, defaultdict
from types import FrameType, TracebackType
from typing import (
from asttokens import ASTText
def cached_property_wrapper(self, obj, _cls):
    if obj is None:
        return self
    value = obj.__dict__[self.func.__name__] = self.func(obj)
    return value